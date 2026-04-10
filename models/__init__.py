"""
Model registry.

To add a new model:
  1. Create models/<your_model>/ with __init__.py, model.py, and any architecture files.
  2. Import the model class here and add an entry to MODEL_REGISTRY.
"""

from .supcon import SupConModel

MODEL_REGISTRY: dict = {
    "supcon": SupConModel,
}
