"""
HRAssist RAG Application - Core Source Package
"""

from src.prompt_engine import HRAssistPromptBuilder, Message, Role, compare_prompt_variations
from src.model_config import ModelConfig, LLMController
from src.document_loader import DocumentLoader, LoadedDocument
from src.document_chunker import DocumentChunker, Chunk, ChunkMetadata, tag_chunks, estimate_tokens
from src.similarity import cosine_similarity, rank_chunks, compare_embeddings
from src.batch_embedding import batches, embed_with_retry, run_batch_embedding

__all__ = [
    "HRAssistPromptBuilder",
    "Message",
    "Role",
    "compare_prompt_variations",
    "ModelConfig",
    "LLMController",
    "DocumentLoader",
    "LoadedDocument",
    "DocumentChunker",
    "Chunk",
    "ChunkMetadata",
    "tag_chunks",
    "estimate_tokens",
    "cosine_similarity",
    "rank_chunks",
    "compare_embeddings",
    "batches",
    "embed_with_retry",
    "run_batch_embedding",
]
