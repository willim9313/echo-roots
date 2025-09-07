"""Taxonomy Management Layer (A Layer) - High-level taxonomy operations.

This module provides comprehensive management for the A-layer taxonomy hierarchy,
including creation, navigation, validation, and governance operations.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from echo_roots.models.taxonomy import Category
from echo_roots.storage.interfaces import TaxonomyRepository


class TaxonomyStats(BaseModel):
    """Statistics for taxonomy structure and health."""
    
    total_categories: int = 0
    max_depth: int = 0
    avg_depth: float = 0.0
    root_categories: int = 0
    leaf_categories: int = 0
    orphaned_categories: int = 0
    category_count_by_level: Dict[int, int] = Field(default_factory=dict)
    domain_coverage: Dict[str, int] = Field(default_factory=dict)


class TaxonomyPath:
    """Utility class for working with taxonomy paths."""
    
    @staticmethod
    def build_path(parent_path: Optional[List[str]], category_name: str) -> List[str]:
        """Build a complete path for a category given its parent path."""
        if parent_path is None:
            return [category_name]
        return parent_path + [category_name]
    
    @staticmethod
    def calculate_level(path: List[str]) -> int:
        """Calculate the hierarchy level from a path."""
        return len(path) - 1
    
    @staticmethod
    def validate_path_consistency(path: List[str], level: int, name: str) -> bool:
        """Validate that path is consistent with level and category name."""
        return (
            len(path) == level + 1 and
            path[-1] == name and
            all(component.strip() for component in path)
        )
    
    @staticmethod
    def get_parent_path(path: List[str]) -> Optional[List[str]]:
        """Get the parent path from a category path."""
        if len(path) <= 1:
            return None
        return path[:-1]


class CategoryCreationRequest(BaseModel):
    """Request model for creating a new category."""
    
    name: str = Field(description="Category name")
    parent_id: Optional[str] = Field(default=None, description="Parent category ID")
    description: Optional[str] = Field(default=None, description="Category description")
    labels: Dict[str, str] = Field(default_factory=dict, description="Multilingual labels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    domain: Optional[str] = Field(default=None, description="Domain context")


class CategoryMoveRequest(BaseModel):
    """Request model for moving a category to a new parent."""
    
    category_id: str = Field(description="Category to move")
    new_parent_id: Optional[str] = Field(description="New parent category ID (None for root)")
    preserve_children: bool = Field(default=True, description="Whether to move children along")


class CategoryMergeRequest(BaseModel):
    """Request model for merging categories."""
    
    source_category_id: str = Field(description="Category to merge from")
    target_category_id: str = Field(description="Category to merge into")
    merge_strategy: str = Field(
        default="replace",
        description="Merge strategy: replace, combine_metadata, prefer_target"
    )


class TaxonomyManager:
    """High-level manager for taxonomy operations and governance.
    
    Provides comprehensive operations for managing the A-layer taxonomy hierarchy
    including creation, navigation, validation, and governance workflows.
    """
    
    def __init__(self, taxonomy_repo: TaxonomyRepository):
        """Initialize the taxonomy manager.
        
        Args:
            taxonomy_repo: Repository for taxonomy operations
        """
        self.taxonomy_repo = taxonomy_repo
        self._category_cache: Dict[str, Category] = {}
        self._path_cache: Dict[str, List[str]] = {}
    
    async def create_category(self, request: CategoryCreationRequest) -> Category:
        """Create a new category with proper hierarchy validation.
        
        Args:
            request: Category creation request
            
        Returns:
            Created category
            
        Raises:
            ValueError: If hierarchy constraints are violated
            ConflictError: If category name already exists in domain
        """
        # Validate parent exists if specified
        parent_category = None
        if request.parent_id:
            parent_category = await self.taxonomy_repo.get_category(request.parent_id)
            if not parent_category:
                raise ValueError(f"Parent category not found: {request.parent_id}")
        
        # Check for name conflicts in domain
        if request.domain:
            existing = await self.taxonomy_repo.get_category_by_name(
                request.name, request.domain
            )
            if existing:
                raise ValueError(
                    f"Category '{request.name}' already exists in domain '{request.domain}'"
                )
        
        # Build path and calculate level
        parent_path = parent_category.path if parent_category else None
        category_path = TaxonomyPath.build_path(parent_path, request.name)
        category_level = TaxonomyPath.calculate_level(category_path)
        
        # Validate depth constraints
        if category_level > 10:
            raise ValueError(f"Maximum taxonomy depth (10) exceeded: {category_level}")
        
        # Create category
        category = Category(
            category_id=str(uuid4()),
            name=request.name,
            parent_id=request.parent_id,
            level=category_level,
            path=category_path,
            description=request.description,
            labels=request.labels,
            metadata=request.metadata or {}
        )
        
        # Store category
        category_id = await self.taxonomy_repo.store_category(category)
        category.category_id = category_id
        
        # Update cache
        self._category_cache[category_id] = category
        self._path_cache[category_id] = category_path
        
        return category
    
    async def get_category(self, category_id: str) -> Optional[Category]:
        """Get a category by ID with caching.
        
        Args:
            category_id: Category identifier
            
        Returns:
            Category if found, None otherwise
        """
        # Check cache first
        if category_id in self._category_cache:
            return self._category_cache[category_id]
        
        # Fetch from repository
        category = await self.taxonomy_repo.get_category(category_id)
        if category:
            self._category_cache[category_id] = category
            self._path_cache[category_id] = category.path
        
        return category
    
    async def get_children(self, category_id: Optional[str] = None) -> List[Category]:
        """Get direct children of a category.
        
        Args:
            category_id: Parent category ID (None for root categories)
            
        Returns:
            List of child categories
        """
        return await self.taxonomy_repo.list_categories(parent_id=category_id)
    
    async def get_descendants(self, category_id: str) -> List[Category]:
        """Get all descendants of a category (recursive).
        
        Args:
            category_id: Root category ID
            
        Returns:
            List of all descendant categories
        """
        descendants = []
        children = await self.get_children(category_id)
        
        for child in children:
            descendants.append(child)
            # Recursively get grandchildren
            grandchildren = await self.get_descendants(child.category_id)
            descendants.extend(grandchildren)
        
        return descendants
    
    async def get_ancestors(self, category_id: str) -> List[Category]:
        """Get all ancestors of a category up to root.
        
        Args:
            category_id: Category ID to find ancestors for
            
        Returns:
            List of ancestor categories (from root to immediate parent)
        """
        category = await self.get_category(category_id)
        if not category or not category.parent_id:
            return []
        
        ancestors = []
        current_parent_id = category.parent_id
        
        while current_parent_id:
            parent = await self.get_category(current_parent_id)
            if not parent:
                break
            
            ancestors.insert(0, parent)  # Insert at beginning for root-to-parent order
            current_parent_id = parent.parent_id
        
        return ancestors
    
    async def get_siblings(self, category_id: str) -> List[Category]:
        """Get sibling categories (same parent).
        
        Args:
            category_id: Category ID to find siblings for
            
        Returns:
            List of sibling categories (excluding the category itself)
        """
        category = await self.get_category(category_id)
        if not category:
            return []
        
        siblings = await self.get_children(category.parent_id)
        return [sibling for sibling in siblings if sibling.category_id != category_id]
    
    async def move_category(self, request: CategoryMoveRequest) -> Category:
        """Move a category to a new parent.
        
        Args:
            request: Move request with category and new parent
            
        Returns:
            Updated category
            
        Raises:
            ValueError: If move would create cycles or violate constraints
        """
        category = await self.get_category(request.category_id)
        if not category:
            raise ValueError(f"Category not found: {request.category_id}")
        
        # Validate new parent exists if specified
        new_parent = None
        if request.new_parent_id:
            new_parent = await self.get_category(request.new_parent_id)
            if not new_parent:
                raise ValueError(f"New parent not found: {request.new_parent_id}")
            
            # Check for circular reference
            ancestors = await self.get_ancestors(request.new_parent_id)
            ancestor_ids = [ancestor.category_id for ancestor in ancestors]
            if request.category_id in ancestor_ids:
                raise ValueError("Cannot move category: would create circular reference")
        
        # Calculate new path and level
        new_parent_path = new_parent.path if new_parent else None
        new_path = TaxonomyPath.build_path(new_parent_path, category.name)
        new_level = TaxonomyPath.calculate_level(new_path)
        
        # Validate depth constraints
        if new_level > 10:
            raise ValueError(f"Move would exceed maximum depth (10): {new_level}")
        
        # Update category
        category.parent_id = request.new_parent_id
        category.level = new_level
        category.path = new_path
        category.updated_at = datetime.now()
        
        # Store updated category
        await self.taxonomy_repo.store_category(category)
        
        # Update children if preserving hierarchy
        if request.preserve_children:
            await self._update_children_paths(category)
        
        # Update cache
        self._category_cache[category.category_id] = category
        self._path_cache[category.category_id] = new_path
        
        return category
    
    async def merge_categories(self, request: CategoryMergeRequest) -> Category:
        """Merge two categories.
        
        Args:
            request: Merge request with source and target categories
            
        Returns:
            Target category with merged data
            
        Raises:
            ValueError: If categories cannot be merged
        """
        source = await self.get_category(request.source_category_id)
        target = await self.get_category(request.target_category_id)
        
        if not source or not target:
            raise ValueError("Both source and target categories must exist")
        
        if source.category_id == target.category_id:
            raise ValueError("Cannot merge category with itself")
        
        # Merge metadata based on strategy
        if request.merge_strategy == "combine_metadata":
            merged_metadata = {**target.metadata, **source.metadata}
        elif request.merge_strategy == "prefer_target":
            merged_metadata = target.metadata
        else:  # replace
            merged_metadata = source.metadata
        
        # Update target category
        target.metadata = merged_metadata
        target.updated_at = datetime.now()
        
        # Merge labels
        merged_labels = {**target.labels, **source.labels}
        target.labels = merged_labels
        
        # Store updated target
        await self.taxonomy_repo.store_category(target)
        
        # Move source children to target
        source_children = await self.get_children(source.category_id)
        for child in source_children:
            await self.move_category(CategoryMoveRequest(
                category_id=child.category_id,
                new_parent_id=target.category_id,
                preserve_children=True
            ))
        
        # Mark source as merged
        source.status = "merged"
        source.metadata["merged_into"] = target.category_id
        source.updated_at = datetime.now()
        await self.taxonomy_repo.store_category(source)
        
        # Update cache
        self._category_cache[target.category_id] = target
        if source.category_id in self._category_cache:
            del self._category_cache[source.category_id]
        
        return target
    
    async def delete_category(self, category_id: str, recursive: bool = False) -> bool:
        """Delete a category.
        
        Args:
            category_id: Category to delete
            recursive: Whether to delete children recursively
            
        Returns:
            True if deleted successfully
            
        Raises:
            ValueError: If category has children and recursive=False
        """
        category = await self.get_category(category_id)
        if not category:
            return False
        
        children = await self.get_children(category_id)
        if children and not recursive:
            raise ValueError(f"Category has {len(children)} children. Use recursive=True to delete all.")
        
        # Recursively delete children if requested
        if recursive:
            for child in children:
                await self.delete_category(child.category_id, recursive=True)
        
        # Mark category as deprecated
        category.status = "deprecated"
        category.updated_at = datetime.now()
        await self.taxonomy_repo.store_category(category)
        
        # Remove from cache
        if category_id in self._category_cache:
            del self._category_cache[category_id]
        if category_id in self._path_cache:
            del self._path_cache[category_id]
        
        return True
    
    async def validate_taxonomy_integrity(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Validate taxonomy structure integrity.
        
        Args:
            domain: Optional domain to validate
            
        Returns:
            Validation report with errors and warnings
        """
        errors = []
        warnings = []
        
        # Get all categories in domain
        categories = await self.taxonomy_repo.list_categories(domain=domain)
        category_dict = {cat.category_id: cat for cat in categories}
        
        for category in categories:
            # Check parent-child consistency
            if category.parent_id:
                if category.parent_id not in category_dict:
                    errors.append(f"Category {category.name} has invalid parent_id: {category.parent_id}")
                else:
                    parent = category_dict[category.parent_id]
                    if category.level != parent.level + 1:
                        errors.append(
                            f"Category {category.name} level {category.level} "
                            f"inconsistent with parent level {parent.level}"
                        )
            
            # Check path consistency
            if not TaxonomyPath.validate_path_consistency(
                category.path, category.level, category.name
            ):
                errors.append(f"Category {category.name} has inconsistent path: {category.path}")
            
            # Check for orphaned categories
            if category.level > 0 and not category.parent_id:
                warnings.append(f"Category {category.name} appears orphaned (level > 0, no parent)")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "is_valid": len(errors) == 0,
            "categories_checked": len(categories)
        }
    
    async def get_taxonomy_stats(self, domain: Optional[str] = None) -> TaxonomyStats:
        """Get comprehensive taxonomy statistics.
        
        Args:
            domain: Optional domain to analyze
            
        Returns:
            Taxonomy statistics and health metrics
        """
        categories = await self.taxonomy_repo.list_categories(domain=domain)
        
        if not categories:
            return TaxonomyStats()
        
        stats = TaxonomyStats()
        stats.total_categories = len(categories)
        
        levels = [cat.level for cat in categories]
        stats.max_depth = max(levels)
        stats.avg_depth = sum(levels) / len(levels)
        
        stats.root_categories = sum(1 for cat in categories if cat.level == 0)
        
        # Count categories per level
        for cat in categories:
            level = cat.level
            stats.category_count_by_level[level] = (
                stats.category_count_by_level.get(level, 0) + 1
            )
        
        # Find leaf categories (no children)
        category_ids = {cat.category_id for cat in categories}
        parent_ids = {cat.parent_id for cat in categories if cat.parent_id}
        leaf_category_ids = category_ids - parent_ids
        stats.leaf_categories = len(leaf_category_ids)
        
        # Count orphaned categories
        for cat in categories:
            if cat.level > 0 and cat.parent_id not in category_ids:
                stats.orphaned_categories += 1
        
        return stats
    
    async def _update_children_paths(self, parent_category: Category) -> None:
        """Update paths for all children after parent move."""
        children = await self.get_children(parent_category.category_id)
        
        for child in children:
            new_path = TaxonomyPath.build_path(parent_category.path, child.name)
            new_level = TaxonomyPath.calculate_level(new_path)
            
            child.path = new_path
            child.level = new_level
            child.updated_at = datetime.now()
            
            await self.taxonomy_repo.store_category(child)
            
            # Update cache
            self._category_cache[child.category_id] = child
            self._path_cache[child.category_id] = new_path
            
            # Recursively update grandchildren
            await self._update_children_paths(child)
    
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._category_cache.clear()
        self._path_cache.clear()
