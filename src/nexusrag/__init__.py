"""NexusRAG: Self-correcting RAG for scientific literature synthesis."""

__version__ = "0.1.0"
__author__ = "Urme Bose"

from nexusrag.config import Settings, get_settings, settings
from nexusrag.pipeline import IngestResult, NexusRAG, SystemStats, get_nexusrag

__all__ = [
    "IngestResult",
    "NexusRAG",
    "Settings",
    "SystemStats",
    "__version__",
    "get_nexusrag",
    "get_settings",
    "settings",
]
