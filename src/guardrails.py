"""Hallucination guardrails and refusal handling for RAG pipelines.

HRS3.41 — Hallucination Guardrails & Refusal Handling

A RAG system should not answer every question. If retrieval is weak, empty,
or returns chunks that score too low to be trusted, the safest response is a
clear refusal rather than a confident answer built on poor evidence.

A hallucination is an unsupported answer that sounds certain. Guardrails
reduce that risk by checking retrieval quality *before* generation so the
model is never asked to work from inadequate context.

This module provides:
  1. ``RetrievalStrengthConfig`` — tuneable thresholds (min score, min chunk
     count) that separate strong retrieval from weak retrieval.
  2. ``retrieval_is_strong()`` — fast boolean gate: passes strong context,
     blocks weak context.
  3. ``assess_retrieval()`` — detailed signal report for debugging and
     threshold tuning.
  4. ``guarded_answer()`` — the full guarded pipeline: retrieve → check →
     refuse or generate.

Threshold guidance
------------------
Start conservative (MIN_TOP_SCORE ≈ 0.72, MIN_SUPPORTING_CHUNKS = 1).
Then measure recall on real queries: lower the threshold if too many valid
questions are refused; raise it if hallucinations still appear. Thresholds
should come from retrieval evaluation data, not guesswork.

Refusing vs answering trade-off
--------------------------------
Refusing too often frustrates users. Answering too freely risks
misinformation. In high-stakes domains (HR policy, legal, health, finance)
refusing is safer when evidence is weak because a confident unsupported
answer can cause real harm.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Refusal constants ─────────────────────────────────────────────────────

REFUSAL_MESSAGE_NO_CONTEXT: str = (
    "I don't have enough reliable context to answer that."
)
REFUSAL_MESSAGE_LOW_SCORE: str = (
    "The retrieved context does not appear closely related to your question. "
    "I don't have enough reliable context to answer that."
)
REFUSAL_MESSAGE_EMPTY: str = (
    "No relevant documents were found for your question. "
    "Please rephrase or contact the HR team directly."
)

STATUS_ANSWERED: str = "answered"
STATUS_REFUSED_WEAK_CONTEXT: str = "refused_weak_context"
STATUS_REFUSED_NO_CHUNKS: str = "refused_no_chunks"


# ── Configuration dataclass ───────────────────────────────────────────────

@dataclass
class RetrievalStrengthConfig:
    """Tuneable thresholds that define what counts as strong retrieval.

    Attributes:
        min_top_score: Minimum cosine similarity score the top-ranked chunk
            must have. Chunks below this are considered unreliable. The
            assignment default is 0.72; tune from retrieval evaluation data.
        min_supporting_chunks: Minimum number of chunks that must meet or
            exceed ``min_top_score``. Setting this to 1 means a single
            high-scoring chunk is sufficient to proceed.
        require_non_empty: When True (the default), an empty chunk list
            always triggers a refusal regardless of thresholds.

    Example::

        config = RetrievalStrengthConfig(min_top_score=0.80,
                                         min_supporting_chunks=2)
        if retrieval_is_strong(chunks, config):
            ...
    """
    min_top_score: float = 0.72
    min_supporting_chunks: int = 1
    require_non_empty: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_top_score <= 1.0):
            raise ValueError(
                f"min_top_score must be in [0.0, 1.0], got {self.min_top_score}"
            )
        if self.min_supporting_chunks < 1:
            raise ValueError(
                f"min_supporting_chunks must be >= 1, "
                f"got {self.min_supporting_chunks}"
            )


# ── Retrieval strength checks ─────────────────────────────────────────────

def retrieval_is_strong(
    chunks: List[Dict[str, Any]],
    config: Optional[RetrievalStrengthConfig] = None,
) -> bool:
    """Return True when retrieved chunks are strong enough to ground an answer.

    Uses simple, fast signals:
      - Empty list → always False (no evidence at all).
      - Count how many chunks score >= ``config.min_top_score``.
      - Return True only when that count >= ``config.min_supporting_chunks``.

    This is the boolean gate used by ``guarded_answer`` before invoking the
    language model. The threshold should be tuned with real queries — start
    conservative (0.72) and lower it only if valid questions are refused.

    Args:
        chunks: Retrieval result dicts, each with a float ``"score"`` key.
        config: Threshold configuration. Defaults to
                ``RetrievalStrengthConfig()`` (min_top_score=0.72,
                min_supporting_chunks=1).

    Returns:
        True when retrieval is strong enough to proceed with generation.

    Example::

        chunks = retrieve(query, corpus, embed_fn, k=4)
        if retrieval_is_strong(chunks):
            answer = generate(question, chunks)
        else:
            answer = REFUSAL_MESSAGE_NO_CONTEXT
    """
    cfg = config or RetrievalStrengthConfig()

    if not chunks:
        return False

    strong_chunks = [
        chunk for chunk in chunks
        if chunk.get("score", 0.0) >= cfg.min_top_score
    ]
    return len(strong_chunks) >= cfg.min_supporting_chunks


def assess_retrieval(
    chunks: List[Dict[str, Any]],
    config: Optional[RetrievalStrengthConfig] = None,
) -> Dict[str, Any]:
    """Return a detailed signal report about retrieval quality.

    Useful for threshold tuning and debugging. Shows the top score, how
    many chunks pass the threshold, and which specific signal caused a
    weak assessment when retrieval fails the gate.

    Args:
        chunks: Retrieval result dicts with ``"score"`` keys.
        config: Threshold configuration (defaults to
                ``RetrievalStrengthConfig()``).

    Returns:
        Dict with keys:
          - ``"is_strong"``            : bool — overall pass/fail
          - ``"total_chunks"``         : int — total chunks received
          - ``"top_score"``            : float — score of the best chunk
                                          (0.0 if no chunks)
          - ``"chunks_above_threshold"``: int — chunks meeting min_top_score
          - ``"min_top_score"``        : float — threshold used
          - ``"min_supporting_chunks`` : int — minimum required
          - ``"failure_reason"``       : str or None — human-readable reason
                                          when is_strong is False

    Example::

        report = assess_retrieval(chunks)
        print(report["top_score"], report["failure_reason"])
    """
    cfg = config or RetrievalStrengthConfig()

    if not chunks:
        return {
            "is_strong": False,
            "total_chunks": 0,
            "top_score": 0.0,
            "chunks_above_threshold": 0,
            "min_top_score": cfg.min_top_score,
            "min_supporting_chunks": cfg.min_supporting_chunks,
            "failure_reason": "no chunks returned by retrieval",
        }

    scores = [chunk.get("score", 0.0) for chunk in chunks]
    top_score = max(scores)
    above = sum(1 for s in scores if s >= cfg.min_top_score)
    strong = above >= cfg.min_supporting_chunks

    failure_reason: Optional[str] = None
    if not strong:
        if top_score < cfg.min_top_score:
            failure_reason = (
                f"top score {top_score:.4f} is below "
                f"min_top_score {cfg.min_top_score}"
            )
        else:
            failure_reason = (
                f"only {above} chunk(s) above threshold, "
                f"need {cfg.min_supporting_chunks}"
            )

    return {
        "is_strong": strong,
        "total_chunks": len(chunks),
        "top_score": round(top_score, 4),
        "chunks_above_threshold": above,
        "min_top_score": cfg.min_top_score,
        "min_supporting_chunks": cfg.min_supporting_chunks,
        "failure_reason": failure_reason,
    }


# ── Guarded answer pipeline ───────────────────────────────────────────────

def guarded_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    generate_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
    config: Optional[RetrievalStrengthConfig] = None,
) -> Dict[str, Any]:
    """Run the full guarded pipeline: check retrieval strength, then generate or refuse.

    Places the guardrail *before* generation so the language model is never
    asked to answer from insufficient evidence. The pipeline:

      1. Check whether ``chunks`` are empty → refuse with ``STATUS_REFUSED_NO_CHUNKS``.
      2. Check whether ``chunks`` meet the strength thresholds → refuse with
         ``STATUS_REFUSED_WEAK_CONTEXT`` if not.
      3. Call ``generate_fn(question, chunks)`` and tag the result with
         ``STATUS_ANSWERED``.

    The ``generate_fn`` is injected so the guardrail layer stays decoupled
    from the specific LLM or prompt builder being used.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks from the vector search step. Each must have
                at least a ``"score"`` key (float).
        generate_fn: Callable ``(question, chunks) -> dict``. Should return a
                     dict with at minimum an ``"answer"`` key and optionally
                     a ``"sources"`` key. The guardrail adds a ``"status"``
                     key to the returned dict.
        config: Threshold configuration. Defaults to
                ``RetrievalStrengthConfig()``.

    Returns:
        Dict with keys:
          - ``"answer"`` : the generated answer string or a refusal message
          - ``"sources"``: list of source metadata dicts (empty on refusal)
          - ``"status"`` : one of ``STATUS_ANSWERED``, ``STATUS_REFUSED_WEAK_CONTEXT``,
                           ``STATUS_REFUSED_NO_CHUNKS``
          - ``"retrieval_assessment"``: the full report from ``assess_retrieval``

    Example::

        def my_generate(question, chunks):
            # call your LLM here
            return {"answer": "...", "sources": [c["metadata"] for c in chunks]}

        result = guarded_answer(
            "How do I apply for sick leave?",
            retrieved_chunks,
            generate_fn=my_generate,
        )
        print(result["status"], result["answer"])
    """
    cfg = config or RetrievalStrengthConfig()
    assessment = assess_retrieval(chunks, cfg)

    # ── Gate 1: no chunks at all ──────────────────────────────────────────
    if not chunks:
        return {
            "answer": REFUSAL_MESSAGE_EMPTY,
            "sources": [],
            "status": STATUS_REFUSED_NO_CHUNKS,
            "retrieval_assessment": assessment,
        }

    # ── Gate 2: chunks exist but score too low ────────────────────────────
    if not assessment["is_strong"]:
        return {
            "answer": REFUSAL_MESSAGE_LOW_SCORE,
            "sources": [],
            "status": STATUS_REFUSED_WEAK_CONTEXT,
            "retrieval_assessment": assessment,
        }

    # ── Generate: retrieval passed both gates ─────────────────────────────
    generated = generate_fn(question, chunks)
    return {
        **generated,
        "status": STATUS_ANSWERED,
        "retrieval_assessment": assessment,
    }


# ── Display helper ────────────────────────────────────────────────────────

def print_guardrail_report(result: Dict[str, Any]) -> None:
    """Print a human-readable guardrail result to stdout.

    Args:
        result: The dict returned by :func:`guarded_answer`.
    """
    print("\n" + "=" * 70)
    print("GUARDRAIL RESULT")
    print("=" * 70)
    print(f"  Status  : {result['status']}")
    print(f"  Answer  : {result['answer'][:200]}")
    print(f"  Sources : {result.get('sources', [])}")

    ra = result.get("retrieval_assessment", {})
    print("\n  Retrieval assessment:")
    print(f"    total_chunks          : {ra.get('total_chunks')}")
    print(f"    top_score             : {ra.get('top_score')}")
    print(f"    chunks_above_threshold: {ra.get('chunks_above_threshold')}")
    print(f"    is_strong             : {ra.get('is_strong')}")
    if ra.get("failure_reason"):
        print(f"    failure_reason        : {ra.get('failure_reason')}")
    print("=" * 70)
