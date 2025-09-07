#!/usr/bin/env python3
"""Setup script for echo-roots development environment."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str) -> None:
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        sys.exit(1)

def main():
    """Set up the development environment."""
    print("🚀 Setting up echo-roots development environment")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run this script from the project root.")
        sys.exit(1)
    
    # Install UV if not present
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing UV package manager...")
        run_command("pip install uv", "Installing UV")
    
    # Create virtual environment and install dependencies
    run_command("uv venv", "Creating virtual environment")
    run_command("uv pip install -e .[dev]", "Installing development dependencies")
    
    # Install pre-commit hooks
    run_command("uv run pre-commit install", "Setting up pre-commit hooks")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Run initial tests to verify setup
    print("🧪 Running initial tests...")
    run_command("uv run pytest tests/ -v --tb=short", "Running test suite")
    
    # Run linting
    print("🔍 Running code quality checks...")
    run_command("uv run ruff check src/ tests/", "Running Ruff linter")
    run_command("uv run ruff format --check src/ tests/", "Checking code formatting")
    
    print("\n🎉 Development environment setup complete!")
    print("\n📋 Next steps:")
    print("   1. Activate the virtual environment: source .venv/bin/activate")
    print("   2. Run tests: uv run pytest")
    print("   3. Start development: uv run echo-roots --help")
    print("   4. View documentation: docs/OVERVIEW.md")

if __name__ == "__main__":
    main()
