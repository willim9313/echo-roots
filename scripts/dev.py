#!/usr/bin/env python3
"""Development utilities for echo-roots project."""

import argparse
import subprocess
import sys
from pathlib import Path

def run_tests(args):
    """Run the test suite with optional filters."""
    cmd = ["uv", "run", "pytest"]
    
    if args.coverage:
        cmd.extend(["--cov=echo_roots", "--cov-report=html", "--cov-report=term"])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
        
    if args.verbose:
        cmd.append("-v")
        
    if args.fast:
        cmd.extend(["-x", "--tb=short"])
        
    if args.path:
        cmd.append(args.path)
    else:
        cmd.append("tests/")
        
    subprocess.run(cmd)

def run_lint(args):
    """Run code quality checks."""
    if args.fix:
        # Auto-fix issues
        subprocess.run(["uv", "run", "ruff", "check", "--fix", "src/", "tests/"])
        subprocess.run(["uv", "run", "ruff", "format", "src/", "tests/"])
    else:
        # Just check
        subprocess.run(["uv", "run", "ruff", "check", "src/", "tests/"])
        subprocess.run(["uv", "run", "ruff", "format", "--check", "src/", "tests/"])

def run_type_check(args):
    """Run type checking with mypy."""
    cmd = ["uv", "run", "mypy"]
    if args.strict:
        cmd.append("--strict")
    cmd.extend(["src/echo_roots"])
    subprocess.run(cmd)

def clean_project(args):
    """Clean build artifacts and cache files."""
    import shutil
    
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo", 
        ".pytest_cache",
        ".coverage",
        "htmlcov/",
        "dist/",
        "build/",
        "*.egg-info/"
    ]
    
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                path.unlink()
                print(f"Removed file: {path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Echo-roots development utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    test_parser.add_argument("--markers", help="Run tests with specific markers")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.add_argument("--fast", "-x", action="store_true", help="Stop on first failure")
    test_parser.add_argument("path", nargs="?", help="Specific test path")
    
    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Run code quality checks")
    lint_parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    
    # Type check command
    type_parser = subparsers.add_parser("typecheck", help="Run type checking")
    type_parser.add_argument("--strict", action="store_true", help="Strict type checking")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean build artifacts")
    
    args = parser.parse_args()
    
    if args.command == "test":
        run_tests(args)
    elif args.command == "lint":
        run_lint(args)
    elif args.command == "typecheck":
        run_type_check(args)
    elif args.command == "clean":
        clean_project(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
