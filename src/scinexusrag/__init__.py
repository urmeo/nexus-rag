"""NexusRAG: local hybrid retrieval for scientific papers."""

__version__ = "1.0.2"
__author__ = "Urme Bose"

from scinexusrag.config import Settings, get_settings, settings
from scinexusrag.pipeline import IngestResult, NexusRAG, SystemStats, get_scinexusrag

__all__ = [
    "IngestResult",
    "NexusRAG",
    "Settings",
    "SystemStats",
    "__version__",
    "get_scinexusrag",
    "get_settings",
    "settings",
]
