"""Data collection package."""
from .dataset_loader import DatasetLoader
from .synthetic_generator import SyntheticDNSGenerator

__all__ = ["DatasetLoader", "SyntheticDNSGenerator"]
