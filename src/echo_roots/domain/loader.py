"""Domain pack loader for YAML configuration files.

This module handles loading, validation, and caching of domain.yaml files.
Domain packs provide flexible configuration for adapting the core framework
to specific domains like e-commerce, media, or knowledge graphs.

Key components:
- DomainPackLoader: Main loader with caching and validation
- load_domain_pack: Convenience function for simple loading
- validate_domain_pack: Standalone validation function
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Union, Any
from functools import lru_cache

from echo_roots.models.domain import DomainPack
from pydantic import ValidationError


class DomainPackLoadError(Exception):
    """Raised when domain pack loading fails."""
    
    def __init__(self, message: str, path: Optional[Path] = None, cause: Optional[Exception] = None):
        self.message = message
        self.path = path
        self.cause = cause
        super().__init__(message)


class DomainPackLoader:
    """Loads and validates domain pack YAML files with caching.
    
    Provides loading, validation, and caching of domain.yaml files.
    Supports both file paths and directory scanning.
    
    Example:
        >>> loader = DomainPackLoader()
        >>> pack = loader.load("domains/ecommerce/domain.yaml")
        >>> print(pack.domain, pack.taxonomy_version)
        
        >>> # Load from directory (looks for domain.yaml)
        >>> pack = loader.load_from_directory("domains/ecommerce")
    """
    
    def __init__(self, cache_size: int = 128, validate_on_load: bool = True):
        """Initialize the domain pack loader.
        
        Args:
            cache_size: Maximum number of packs to cache in memory
            validate_on_load: Whether to validate packs during loading
        """
        self.cache_size = cache_size
        self.validate_on_load = validate_on_load
        self._cache: Dict[str, DomainPack] = {}
    
    def load(self, path: Union[str, Path]) -> DomainPack:
        """Load a domain pack from a YAML file.
        
        Args:
            path: Path to the domain.yaml file
            
        Returns:
            Loaded and validated DomainPack
            
        Raises:
            DomainPackLoadError: If loading or validation fails
        """
        path = Path(path).resolve()
        
        # Check cache first
        cache_key = str(path)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Load YAML file
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            if not isinstance(raw_data, dict):
                raise DomainPackLoadError(
                    f"Domain pack must be a YAML object, got {type(raw_data).__name__}",
                    path
                )
            
            # Create DomainPack (with validation if enabled)
            if self.validate_on_load:
                pack = DomainPack(**raw_data)
            else:
                # Basic validation only
                pack = DomainPack.model_construct(**raw_data)
            
            # Cache the result
            if len(self._cache) >= self.cache_size:
                # Simple LRU: remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = pack
            return pack
            
        except FileNotFoundError:
            raise DomainPackLoadError(f"Domain pack file not found: {path}", path)
        except yaml.YAMLError as e:
            raise DomainPackLoadError(f"Invalid YAML syntax: {e}", path, e)
        except ValidationError as e:
            raise DomainPackLoadError(f"Domain pack validation failed: {e}", path, e)
        except Exception as e:
            raise DomainPackLoadError(f"Failed to load domain pack: {e}", path, e)
    
    def load_from_directory(self, directory: Union[str, Path]) -> DomainPack:
        """Load a domain pack from a directory (looks for domain.yaml).
        
        Args:
            directory: Path to directory containing domain.yaml
            
        Returns:
            Loaded and validated DomainPack
            
        Raises:
            DomainPackLoadError: If domain.yaml not found or invalid
        """
        directory = Path(directory)
        domain_file = directory / "domain.yaml"
        
        if not domain_file.exists():
            # Also try domain.yml
            domain_file = directory / "domain.yml"
            if not domain_file.exists():
                raise DomainPackLoadError(
                    f"No domain.yaml or domain.yml found in {directory}",
                    directory
                )
        
        return self.load(domain_file)
    
    def reload(self, path: Union[str, Path]) -> DomainPack:
        """Reload a domain pack, bypassing cache.
        
        Args:
            path: Path to the domain.yaml file
            
        Returns:
            Freshly loaded DomainPack
        """
        path = Path(path).resolve()
        cache_key = str(path)
        
        # Remove from cache if present
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return self.load(path)
    
    def clear_cache(self) -> None:
        """Clear the domain pack cache."""
        self._cache.clear()
    
    def list_cached(self) -> Dict[str, str]:
        """List currently cached domain packs.
        
        Returns:
            Dict mapping file paths to domain names
        """
        return {path: pack.domain for path, pack in self._cache.items()}


def load_domain_pack(path: Union[str, Path], validate: bool = True) -> DomainPack:
    """Load a domain pack from a YAML file (convenience function).
    
    Args:
        path: Path to the domain.yaml file
        validate: Whether to validate the pack during loading
        
    Returns:
        Loaded DomainPack
        
    Raises:
        DomainPackLoadError: If loading or validation fails
        
    Example:
        >>> pack = load_domain_pack("domains/ecommerce/domain.yaml")
        >>> print(f"Loaded {pack.domain} v{pack.taxonomy_version}")
    """
    loader = DomainPackLoader(validate_on_load=validate)
    return loader.load(path)


def validate_domain_pack(data: Dict[str, Any]) -> DomainPack:
    """Validate raw domain pack data.
    
    Args:
        data: Raw domain pack data (from YAML)
        
    Returns:
        Validated DomainPack
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> with open("domain.yaml") as f:
        ...     raw_data = yaml.safe_load(f)
        >>> pack = validate_domain_pack(raw_data)
    """
    return DomainPack(**data)


@lru_cache(maxsize=32)
def load_cached_domain_pack(path: str) -> DomainPack:
    """Load a domain pack with simple LRU caching.
    
    Args:
        path: Path to the domain.yaml file (as string for hashing)
        
    Returns:
        Cached DomainPack
        
    Note:
        This is a simple caching function. For more control,
        use DomainPackLoader directly.
    """
    return load_domain_pack(path)


def scan_domain_packs(base_directory: Union[str, Path]) -> Dict[str, DomainPack]:
    """Scan a directory tree for domain packs.
    
    Args:
        base_directory: Root directory to scan
        
    Returns:
        Dict mapping domain names to loaded DomainPacks
        
    Raises:
        DomainPackLoadError: If any domain pack fails to load
        
    Example:
        >>> packs = scan_domain_packs("domains/")
        >>> print(f"Found domains: {list(packs.keys())}")
    """
    base_path = Path(base_directory)
    loader = DomainPackLoader()
    packs = {}
    
    # Look for domain.yaml files recursively
    for yaml_file in base_path.rglob("domain.yaml"):
        try:
            pack = loader.load(yaml_file)
            packs[pack.domain] = pack
        except DomainPackLoadError:
            # Also try domain.yml
            yml_file = yaml_file.with_suffix('.yml')
            if yml_file.exists():
                pack = loader.load(yml_file)
                packs[pack.domain] = pack
            else:
                raise
    
    # Also scan for domain.yml files
    for yml_file in base_path.rglob("domain.yml"):
        yaml_equivalent = yml_file.with_suffix('.yaml')
        if yaml_equivalent.exists():
            continue  # Already processed the .yaml version
        
        pack = loader.load(yml_file)
        if pack.domain not in packs:  # Don't override .yaml files
            packs[pack.domain] = pack
    
    return packs


def get_domain_pack_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic info about a domain pack without full validation.
    
    Args:
        path: Path to the domain.yaml file
        
    Returns:
        Dict with basic domain pack information
        
    Raises:
        DomainPackLoadError: If file cannot be read
        
    Example:
        >>> info = get_domain_pack_info("domains/ecommerce/domain.yaml")
        >>> print(f"Domain: {info['domain']}, Version: {info['taxonomy_version']}")
    """
    path = Path(path)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
        
        if not isinstance(raw_data, dict):
            raise DomainPackLoadError(f"Invalid domain pack format in {path}")
        
        return {
            'domain': raw_data.get('domain', 'unknown'),
            'taxonomy_version': raw_data.get('taxonomy_version', 'unknown'),
            'attributes': len(raw_data.get('output_schema', {}).get('attributes', [])),
            'prompts': list(raw_data.get('prompts', {}).keys()),
            'file_path': str(path),
            'file_size': path.stat().st_size,
        }
        
    except Exception as e:
        raise DomainPackLoadError(f"Failed to read domain pack info: {e}", path, e)
