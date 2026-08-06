"""Persian RAG for the Iranian Labor Law."""

from .config import RAGConfig, Settings
from .models import AnswerStatus, Citation, RAGResult
from .pipeline import RAGPipeline

__all__ = [
    "AnswerStatus",
    "Citation",
    "RAGConfig",
    "RAGPipeline",
    "RAGResult",
    "Settings",
]

__version__ = "1.0.0"
