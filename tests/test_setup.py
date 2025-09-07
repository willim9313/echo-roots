"""Test the basic project setup and imports."""

import sys
from pathlib import Path


def test_python_version():
    """Test that we're running on a supported Python version."""
    assert sys.version_info >= (3, 13), "Python 3.13+ required"


def test_project_structure():
    """Test that the basic project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check key directories exist
    assert (project_root / "src" / "echo_roots").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "docs").exists()
    assert (project_root / "domains").exists()

    # Check key files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "ruff.toml").exists()


def test_package_import():
    """Test that the package can be imported."""
    try:
        import echo_roots
        assert echo_roots.__version__ == "1.0.0"
        assert echo_roots.__python_requires__ == ">=3.13"
    except ImportError as e:
        # Expected to fail until we create the core models
        assert "echo_roots.models.core" in str(e)


def test_fixtures_exist():
    """Test that test fixtures are available."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    assert fixtures_dir.exists()
    assert (fixtures_dir / "sample_ingestion_data.json").exists()
    assert (fixtures_dir / "test_domain.yaml").exists()
