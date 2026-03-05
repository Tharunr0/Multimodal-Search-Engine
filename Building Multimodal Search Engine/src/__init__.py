# src/__init__.py

from .embedder import MultimodalEmbedder
from .search import VectorStore
from .data_loader import get_coco_data

# Setting __all__ tells Python what to export when someone runs 'from src import *'
__all__ = [
    "MultimodalEmbedder",
    "VectorStore",
    "get_coco_data"
]

__version__ = "1.0.0"
__author__ = "Your Name"