"""
pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))
