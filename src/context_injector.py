"""Context injection and prompt augmentation for RAG pipelines.

HRS3.38 — Context Injection & Prompt Augmentation

Retrieval gives you chunks. Prompt augmentation turns those chunks into a
grounded prompt the model can actually use. This module handles:

  1. Chunk formatting  — label each retrieved chunk with a source marker
                         so answers can be cited back to their origin.
  2. Token budgeting   — assemble context greedily, stopping before the
                         budget is exceeded, so the full prompt always fits
                         within the model's context window.
  3. Prompt assembly   — combine system instructions, formatted context,
                         and the user question into a single structured prompt
                         that explicitly instructs the model to answer only
                         from the provided evidence.
  4. Overflow handling — when chunks exceed the budget, strategies like
                         truncation and summarisation keep the most useful
                         evidence while respecting token limits.

This is the control point where you decide what evidence the model sees,
how sources are named, and what to do when there is not enough evidence.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.token_counter import count_tokens


# ── Token budget defaults ─────────────────────────────────────────────────
# A typical 8 000-token context window breakdown:
#   instructions + question  ~  800 tokens  (RESERVED_FOR_INSTRUCTIONS)
#   model answer             ~ 1 500 tokens  (RESERVED_FOR_ANSWER)
#   retrieved context        ~ 5 700 tokens  (what remains for chunks)
# Override these constants or pass explicit budgets to the functions below.

DEFAULT_MAX_CONTEXT_TOKENS: int = 5_000
DEFAULT_RESERVED_FOR_ANSWER: int = 1_500
DEFAULT_RESERVED_FOR_INSTRUCTIONS: int = 800


# ── Source marker & chunk formatting ─────────────────────────────────────

def format_chunk(index: int, chunk: Dict[str, Any]) -> str:
    """Format a retrieved chunk with a numbered source marker.

    The marker ``[N] source#chunk_index`` lets the model (and the human
    reviewer) trace every claim in the generated answer back to its origin.
    Without source markers, the model can use context but cannot clearly
    point back to where the evidence came from.

    Args:
        index: 1-based position in the assembled context. Used as the
               citation label ``[1]``, ``[2]``, etc.
        chunk: A retrieval result dict with at minimum:
               - ``"text"``     : the raw chunk text
               - ``"metadata"`` : dict containing at least ``"source"``

    Returns:
        A formatted string::

            [1] employee_leave_policy.txt#2
            Employees must submit leave requests through the HR portal...

    Example::

        formatted = format_chunk(1, {
            "text": "Sick leave requires a medical certificate.",
            "metadata": {"source": "policy.txt", "chunk_index": 3}
        })
    """
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index")

    if chunk_index is not None:
        marker = f"[{index}] {source}#{chunk_index}"
    else:
        marker = f"[{index}] {source}"

    return f"{marker}\n{chunk.get('text', '')}"


# ── Token-budgeted context assembly ──────────────────────────────────────

def assemble_context(
    chunks: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> Tuple[str, int, int]:
    """Greedily assemble formatted chunks up to a token budget.

    Iterates through chunks in the order given (best-ranked first), formats
    each one, and adds it to the context until the next chunk would exceed
    ``max_tokens``. This ensures the assembled context always fits within
    the model's context window without truncating individual chunks
    mid-sentence.

    The caller should pass chunks already sorted by relevance (highest
    first) so the most useful evidence is always included when the budget
    runs out.

    Args:
        chunks: Retrieval result dicts, sorted by descending relevance.
                Each must have ``"text"`` and ``"metadata"`` keys.
        max_tokens: Maximum total tokens for the assembled context string.
                    Must be >= 1.

    Returns:
        A 3-tuple of ``(context_str, used_tokens, chunks_included)`` where:
          - ``context_str``     : chunks joined by ``"\\n\\n---\\n\\n"``
          - ``used_tokens``     : exact token count of ``context_str``
          - ``chunks_included`` : number of chunks that fit in the budget

    Raises:
        ValueError: If ``max_tokens`` < 1.

    Example::

        context, tokens, n = assemble_context(ranked_chunks, max_tokens=5000)
        print(f"Assembled {n} chunks using {tokens} tokens")
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

    selected: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk)
        # Compute token cost of the full joined string if we add this chunk
        candidate = "\n\n---\n\n".join(selected + [formatted])
        if count_tokens(candidate) > max_tokens:
            break
        selected.append(formatted)

    context_str = "\n\n---\n\n".join(selected)
    used_tokens = count_tokens(context_str) if context_str else 0
    return context_str, used_tokens, len(selected)


def assemble_context_with_truncation(
    chunks: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    truncation_suffix: str = "... [truncated]",
) -> Tuple[str, int, int]:
    """Assemble context, truncating the last chunk if it partially fits.

    Unlike ``assemble_context``, this function will include a partially-
    fitting chunk by truncating its text so it fits within the budget.
    This squeezes more evidence into the window when a single large chunk
    would otherwise be completely dropped.

    Args:
        chunks: Retrieval result dicts, sorted by descending relevance.
        max_tokens: Maximum total tokens for the assembled context.
        truncation_suffix: Appended to truncated text to signal the cut.

    Returns:
        Same 3-tuple as ``assemble_context``.

    Raises:
        ValueError: If ``max_tokens`` < 1.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

    selected: List[str] = []
    used_tokens: int = 0

    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk)
        separator_tokens = count_tokens("\n\n---\n\n") if selected else 0
        token_count = count_tokens(formatted)
        remaining = max_tokens - used_tokens - separator_tokens

        if remaining <= 0:
            break

        if token_count <= remaining:
            # Chunk fits whole
            selected.append(formatted)
            used_tokens += separator_tokens + token_count
        else:
            # Truncate to fit the remaining budget
            words = formatted.split()
            truncated = ""
            for word_count in range(len(words), 0, -1):
                candidate = " ".join(words[:word_count]) + truncation_suffix
                if count_tokens(candidate) <= remaining:
                    truncated = candidate
                    break
            if truncated:
                selected.append(truncated)
                used_tokens += separator_tokens + count_tokens(truncated)
            break

    context_str = "\n\n---\n\n".join(selected)
    return context_str, used_tokens, len(selected)


# ── Augmented prompt builder ──────────────────────────────────────────────

# System instruction template used by build_augmented_prompt().
# The "only from context" clause is the grounding guarantee: it instructs
# the model to stay within the retrieved evidence even though the model
# may know general facts about the topic.
_GROUNDED_SYSTEM_INSTRUCTION = (
    "You are a grounded assistant. "
    "Answer the question using ONLY the provided context. "
    'If the answer is not in the context, say: '
    '"I don\'t have enough information in the provided context." '
    "When possible, cite sources using the markers like [1] or [2]."
)


def build_augmented_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    system_instruction: Optional[str] = None,
    use_truncation: bool = False,
) -> Dict[str, Any]:
    """Build a grounded, token-budgeted prompt from retrieved chunks.

    This is the central function for prompt augmentation. It:
      1. Formats each chunk with a source marker.
      2. Assembles chunks into a context string within the token budget.
      3. Combines system instructions, context, and question into a prompt.
      4. Returns metadata (token counts, sources used, chunks dropped) for
         auditing and debugging.

    The assembled prompt format::

        <system_instruction>

        Context:
        [1] source.txt#0
        chunk text here...

        ---

        [2] other.md#1
        more text here...

        Question: <question>

    Args:
        question: The user's natural-language question.
        retrieved_chunks: Retrieval result dicts, sorted by relevance.
                          Each must have ``"text"`` and ``"metadata"`` keys.
        max_context_tokens: Token budget for the assembled context section.
                            Does not include system instructions or question.
        system_instruction: Override the default grounded system instruction.
                            Pass ``None`` to use the built-in grounding prompt.
        use_truncation: When True, uses ``assemble_context_with_truncation``
                        so the last chunk is truncated rather than dropped.

    Returns:
        Dict with keys:
          - ``"prompt"``            : the full assembled prompt string
          - ``"messages"``          : OpenAI-style list of message dicts
                                      ``[{"role": "system", "content": ...},
                                         {"role": "user",   "content": ...}]``
          - ``"context_tokens"``    : tokens used by the context section
          - ``"total_prompt_tokens"``: tokens in the complete prompt string
          - ``"chunks_included"``   : number of chunks that fit in the budget
          - ``"chunks_dropped"``    : number of chunks excluded by the budget
          - ``"sources_used"``      : list of metadata dicts for included chunks

    Raises:
        ValueError: If ``max_context_tokens`` < 1.

    Example::

        result = build_augmented_prompt(
            "How do I apply for sick leave?",
            ranked_chunks,
            max_context_tokens=4000,
        )
        print(result["prompt"])
        print("sources:", result["sources_used"])
    """
    if max_context_tokens < 1:
        raise ValueError(f"max_context_tokens must be >= 1, got {max_context_tokens}")

    instruction = system_instruction or _GROUNDED_SYSTEM_INSTRUCTION

    # Assemble the context section within the token budget
    assemble_fn = (
        assemble_context_with_truncation if use_truncation else assemble_context
    )
    context_str, context_tokens, chunks_included = assemble_fn(
        retrieved_chunks, max_tokens=max_context_tokens
    )

    chunks_dropped = len(retrieved_chunks) - chunks_included

    # Build the full prompt string
    if context_str:
        prompt = (
            f"{instruction}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}"
        )
    else:
        # No chunks fit within the budget — still answer but warn the model
        prompt = (
            f"{instruction}\n\n"
            "Context: (no context available within token budget)\n\n"
            f"Question: {question}"
        )

    # OpenAI-style messages structure
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_str}\n\nQuestion: {question}"
                if context_str
                else f"Question: {question}"
            ),
        },
    ]

    # Sources used = metadata for chunks that were included
    sources_used = [
        retrieved_chunks[i].get("metadata", {})
        for i in range(chunks_included)
    ]

    return {
        "prompt": prompt,
        "messages": messages,
        "context_tokens": context_tokens,
        "total_prompt_tokens": count_tokens(prompt),
        "chunks_included": chunks_included,
        "chunks_dropped": chunks_dropped,
        "sources_used": sources_used,
    }


# ── Display helper ────────────────────────────────────────────────────────

def print_prompt_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of a ``build_augmented_prompt`` result.

    Args:
        result: The dict returned by :func:`build_augmented_prompt`.
    """
    print("\n" + "=" * 70)
    print("AUGMENTED PROMPT SUMMARY")
    print("=" * 70)
    print(f"  Chunks included    : {result['chunks_included']}")
    print(f"  Chunks dropped     : {result['chunks_dropped']}")
    print(f"  Context tokens     : {result['context_tokens']}")
    print(f"  Total prompt tokens: {result['total_prompt_tokens']}")
    print(f"  Sources used       :")
    for meta in result["sources_used"]:
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "n/a")
        print(f"    - {source}#{chunk_idx}")
    print("=" * 70)
    print("\nPROMPT:")
    print("-" * 70)
    print(result["prompt"])
    print("=" * 70)
