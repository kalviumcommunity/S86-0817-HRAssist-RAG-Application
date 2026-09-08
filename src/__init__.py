"""
HRAssist RAG Application - Core Source Package
"""

from src.prompt_engine import HRAssistPromptBuilder, Message, Role, compare_prompt_variations
from src.model_config import ModelConfig, LLMController
from src.document_loader import DocumentLoader, LoadedDocument
from src.document_chunker import DocumentChunker, Chunk, ChunkMetadata, tag_chunks, estimate_tokens
from src.similarity import cosine_similarity, rank_chunks, compare_embeddings
from src.indexer import to_vector_record, index_embeddings, spot_check
from src.batch_embedding import batches, embed_with_retry, run_batch_embedding
from src.embedding_quality import (
    run_sanity_checks,
    build_sanity_report,
    check_dimension_consistency,
    detect_near_duplicate_chunks,
    DEFAULT_TEST_CASES,
)
from src.retriever import retrieve, retrieve_at_k_values
from src.retrieval_tuning import evaluate_retrieval_settings, choose_best_setting
from src.rag_pipeline import (
    embed_query,
    retrieve_context,
    assemble_retrieved_context,
    answer_query,
)
from src.citations import (
    build_citation_map,
    build_cited_prompt,
    cited_markers,
    validate_citations,
    verify_citation,
)
from src.reranker import (
    keyword_overlap_score,
    rerank,
    rerank_with_llm,
    build_reranking_report,
)
from src.context_injector import (
    format_chunk,
    assemble_context,
    assemble_context_with_truncation,
    build_augmented_prompt,
)
from src.guardrails import (
    RetrievalStrengthConfig,
    retrieval_is_strong,
    assess_retrieval,
    guarded_answer,
)
from src.conversational_rag import (
    ConversationHistory,
    rewrite_followup,
    rewrite_followup_simple,
    conversational_answer,
)
from src.document_processor import (
    validate_upload,
    store_upload,
    process_uploaded_document,
    UploadValidationError,
    VECTOR_STORE,
)
from src.observability import (
    cache_key,
    get_cached_answer,
    save_cached_answer,
    invalidate_cache,
    cache_size,
    estimate_cost,
    build_usage_metadata,
    log_rag_request,
    get_usage_log,
    clear_usage_log,
    summarize_usage,
    new_request_id,
)

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
    "to_vector_record",
    "index_embeddings",
    "spot_check",
    "batches",
    "embed_with_retry",
    "run_batch_embedding",
    "run_sanity_checks",
    "build_sanity_report",
    "check_dimension_consistency",
    "detect_near_duplicate_chunks",
    "DEFAULT_TEST_CASES",
    "retrieve",
    "retrieve_at_k_values",
    "evaluate_retrieval_settings",
    "choose_best_setting",
    "embed_query",
    "retrieve_context",
    "assemble_retrieved_context",
    "answer_query",
    "build_citation_map",
    "build_cited_prompt",
    "cited_markers",
    "validate_citations",
    "verify_citation",
    "keyword_overlap_score",
    "rerank",
    "rerank_with_llm",
    "build_reranking_report",
    "format_chunk",
    "assemble_context",
    "assemble_context_with_truncation",
    "build_augmented_prompt",
    "RetrievalStrengthConfig",
    "retrieval_is_strong",
    "assess_retrieval",
    "guarded_answer",
    "ConversationHistory",
    "rewrite_followup",
    "rewrite_followup_simple",
    "conversational_answer",
    "validate_upload",
    "store_upload",
    "process_uploaded_document",
    "UploadValidationError",
    "VECTOR_STORE",
    "cache_key",
    "get_cached_answer",
    "save_cached_answer",
    "invalidate_cache",
    "cache_size",
    "estimate_cost",
    "build_usage_metadata",
    "log_rag_request",
    "get_usage_log",
    "clear_usage_log",
    "summarize_usage",
    "new_request_id",
]
