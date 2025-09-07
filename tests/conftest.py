"""Test configuration and shared fixtures for echo-roots tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ecommerce_data() -> list[dict[str, Any]]:
    """Sample e-commerce data for testing pipelines."""
    return [
        {
            "product_name": "Samsung Galaxy S24 Ultra",
            "desc": "Flagship Android smartphone with S Pen",
            "brand": "Samsung",
            "color": "Titanium Black",
            "storage": "512GB",
            "price": 1199.99,
            "category": "Electronics/Mobile Phones",
            "lang": "en"
        },
        {
            "product_name": "Sony WH-1000XM5",
            "desc": "Premium noise-canceling headphones",
            "brand": "Sony",
            "color": "Black",
            "type": "Over-ear",
            "price": 399.99,
            "category": "Electronics/Audio",
            "lang": "en"
        },
        {
            "title": "Nike Air Force 1",
            "description": "Classic white leather sneakers",
            "brand": "Nike",
            "color": "White",
            "size": "US 10",
            "price": 90.00,
            "category": "Fashion/Footwear",
            "language": "en"
        }
    ]
