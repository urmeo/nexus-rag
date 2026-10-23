import threading

from scinexusrag.application import NexusRAG
from scinexusrag.settings import Settings

_instance: NexusRAG | None = None
_instance_lock = threading.Lock()


def get_scinexusrag(settings: Settings | None = None) -> NexusRAG:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NexusRAG(settings)
    return _instance
