"""Conversational RAG with follow-up query rewriting.

HRS3.42 — Conversational RAG & Follow-Up Context

Real users ask follow-up questions: "What about the deadline?", "Can you
explain that?", "Does it apply to Sprint 2?" Those questions are natural in
conversation but weak for retrieval because they depend on earlier turns.

The solution is a two-step pattern:
  1. **Rewrite** — use the conversation history to turn the follow-up into
     a self-contained standalone query before retrieval.
  2. **Retrieve** — embed and search with the rewritten query, which carries
     enough context to find the right chunks even when the original question
     uses pronouns or references earlier answers.

Token budget for history
------------------------
Sending unlimited history to the rewriter wastes tokens and can confuse
generation. Keep a short rolling window (``max_history_turns``) and
optionally summarise older turns. The history is used only to resolve
references — never as the primary source of truth. Always retrieve fresh
context for each turn.

Critical distinction
--------------------
  - The **rewritten query** is used for *retrieval* (embedding search).
  - The **original user question** is used for *answer generation* so the
    response reads naturally rather than as a paraphrase of a search query.
"""

from typing import Any, Callable, Dict, List, Optional

from src.token_counter import count_tokens
from src.guardrails import (
    retrieval_is_strong,
    RetrievalStrengthConfig,
    REFUSAL_MESSAGE_NO_CONTEXT,
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
    STATUS_REFUSED_NO_CHUNKS,
)


# ── Conversation history management ───────────────────────────────────────

class ConversationHistory:
    """Rolling conversation history with a token-bounded window.

    Stores user and assistant turns as OpenAI-style message dicts and
    exposes a trimmed view that fits within a token budget. This prevents
    old turns from overflowing the context window and keeps the rewriter
    focused on the most recent, relevant dialogue.

    Args:
        max_turns: Maximum number of *complete* turns (user + assistant pairs)
                   to keep in the active window. Older turns are dropped
                   when the window is full.
        max_tokens: If set, the history passed to the rewriter is further
                    trimmed until it fits within this token count.

    Example::

        history = ConversationHistory(max_turns=3)
        history.add_user("What is the sick leave policy?")
        history.add_assistant("Employees get 10 days per year.")
        history.add_user("What about maternity leave?")
        print(history.messages)
    """

    def __init__(
        self,
        max_turns: int = 5,
        max_tokens: Optional[int] = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._messages: List[Dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Append a user turn to the history."""
        self._messages.append({"role": "user", "content": content})
        self._enforce_window()

    def add_assistant(self, content: str) -> None:
        """Append an assistant turn to the history."""
        self._messages.append({"role": "assistant", "content": content})
        self._enforce_window()

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """Append a complete user + assistant turn at once."""
        self._messages.append({"role": "user",      "content": user_content})
        self._messages.append({"role": "assistant", "content": assistant_content})
        self._enforce_window()

    @property
    def messages(self) -> List[Dict[str, str]]:
        """Return the full current message list."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Return the number of complete user+assistant pairs stored."""
        return len(self._messages) // 2

    def token_count(self) -> int:
        """Return the total token count of all stored messages."""
        return sum(count_tokens(m["content"]) for m in self._messages)

    def trimmed_messages(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Return history messages that fit within *max_tokens*.

        Drops the oldest messages (in pairs where possible) until the total
        token count is within the budget. Uses ``self._max_tokens`` when
        *max_tokens* is None; returns all messages when neither is set.

        Args:
            max_tokens: Override the instance-level token budget.

        Returns:
            List of message dicts, newest-biased.
        """
        budget = max_tokens or self._max_tokens
        if budget is None:
            return list(self._messages)

        msgs = list(self._messages)
        while msgs and sum(count_tokens(m["content"]) for m in msgs) > budget:
            msgs.pop(0)

        return msgs

    def clear(self) -> None:
        """Remove all stored turns."""
        self._messages.clear()

    # ── Private helpers ───────────────────────────────────────────────────

    def _enforce_window(self) -> None:
        """Drop oldest messages to stay within max_turns pairs."""
        max_messages = self._max_turns * 2
        while len(self._messages) > max_messages:
            self._messages.pop(0)


# ── Follow-up query rewriting ─────────────────────────────────────────────

# Default system instruction for the rewriter LLM call.
_REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriter. Your only job is to rewrite the user's latest "
    "question as a standalone search query that can be understood without any "
    "conversation history. Use the history only to resolve pronouns and "
    "references. Do NOT answer the question — output only the rewritten query."
)


def rewrite_followup(
    history: List[Dict[str, str]],
    question: str,
    llm_client: Any,
    model: str,
    max_history_tokens: int = 800,
) -> str:
    """Rewrite a follow-up question into a standalone retrieval query.

    Follow-up questions like "What about the video?" are ambiguous for
    embedding search. This function sends a small, bounded slice of the
    conversation history to an LLM with a single instruction: turn the
    follow-up into a self-contained search string.

    The rewritten query is used for retrieval only. The original user
    question is still used for the final answer so the response reads
    naturally.

    Args:
        history: Recent conversation turns as OpenAI-style message dicts.
                 Typically the output of ``ConversationHistory.trimmed_messages()``.
        question: The user's latest follow-up question.
        llm_client: An OpenAI-compatible client with
                    ``chat.completions.create``.
        model: Chat model identifier.
        max_history_tokens: Maximum token budget for the history section of
                            the rewrite prompt. Older messages are dropped
                            first when the history exceeds this budget.

    Returns:
        A rewritten standalone query string. Falls back to the original
        question if the LLM call fails or returns an empty string.

    Example::

        history = [
            {"role": "user",      "content": "What evidence is needed for submission?"},
            {"role": "assistant", "content": "You need a PR link and a video."},
        ]
        query = rewrite_followup(history, "What about the video?", client, model)
        # → "What video explanation is required for project submission?"
    """
    # Trim history to fit the token budget
    trimmed: List[Dict[str, str]] = []
    running_tokens = 0
    for msg in reversed(history):
        t = count_tokens(msg["content"])
        if running_tokens + t > max_history_tokens:
            break
        trimmed.insert(0, msg)
        running_tokens += t

    # Format history as a plain text block for the prompt
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in trimmed
    ) if trimmed else "(no prior conversation)"

    user_prompt = (
        f"History:\n{history_text}\n\n"
        f"Latest question: {question}\n\n"
        "Rewritten standalone query:"
    )

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        # Graceful fallback: use the original question as the query
        return question


def rewrite_followup_simple(
    history: List[Dict[str, str]],
    question: str,
) -> str:
    """Heuristic follow-up rewriter that requires no LLM call.

    Extracts nouns and key phrases from the most recent assistant turn and
    prepends them to the question if the question looks like a follow-up
    (starts with a pronoun, "what about", "does it", etc.). Useful as a
    zero-cost fallback or for testing without an API key.

    Args:
        history: Recent conversation turns.
        question: The user's latest question.

    Returns:
        A slightly expanded query string, or the original if no follow-up
        signals are detected.
    """
    FOLLOWUP_TRIGGERS = {
        "what about", "does it", "can you", "how about", "explain that",
        "tell me more", "and the", "what is the",
    }
    q_lower = question.lower().strip()

    # Check whether the question looks like a follow-up
    is_followup = (
        q_lower.startswith(("it ", "they ", "that ", "this ", "those ", "these "))
        or any(q_lower.startswith(trigger) for trigger in FOLLOWUP_TRIGGERS)
    )

    if not is_followup or not history:
        return question

    # Pull content words from the last assistant turn as extra context
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"),
        "",
    )
    # Take the first 8 words of the last assistant reply as context hint
    context_hint = " ".join(last_assistant.split()[:8])
    if context_hint:
        return f"{question} (context: {context_hint})"
    return question


# ── Conversational answer pipeline ────────────────────────────────────────

def conversational_answer(
    question: str,
    history: ConversationHistory,
    retrieve_fn: Callable[[str], List[Dict[str, Any]]],
    generate_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
    rewrite_fn: Optional[Callable[[List[Dict[str, str]], str], str]] = None,
    guardrail_config: Optional[RetrievalStrengthConfig] = None,
    update_history: bool = True,
) -> Dict[str, Any]:
    """Run one turn of conversational RAG with follow-up rewriting.

    Pipeline for each turn:
      1. Rewrite the user question into a standalone query using history.
      2. Retrieve chunks with the rewritten query.
      3. Check retrieval strength (guardrail).
      4. Generate an answer from the original question + retrieved context.
      5. Append user question and assistant answer to history.

    The rewritten query goes to retrieval; the original user question goes
    to generation so the answer reads naturally, not like a search string.

    Args:
        question: The user's latest question (may be a follow-up).
        history: ``ConversationHistory`` instance tracking the dialogue.
        retrieve_fn: Callable ``(query: str) -> List[chunk_dicts]``.
                     Returns retrieval result dicts with at least ``"score"``,
                     ``"text"``, and ``"metadata"`` keys.
        generate_fn: Callable ``(question: str, chunks: List) -> dict``.
                     Returns a dict with at minimum an ``"answer"`` key.
        rewrite_fn: Optional callable ``(history_msgs, question) -> str``.
                    When None, the original question is used as-is (no
                    rewriting). Pass ``rewrite_followup`` (bound to a client
                    and model) or ``rewrite_followup_simple``.
        guardrail_config: Threshold config for ``retrieval_is_strong``.
                          Defaults to ``RetrievalStrengthConfig()``.
        update_history: When True (default), appends the question and answer
                        to ``history`` after generation.

    Returns:
        Dict with keys:
          - ``"answer"``          : the generated answer or a refusal message
          - ``"rewritten_query"`` : the standalone query used for retrieval
          - ``"sources"``         : list of metadata dicts from retrieved chunks
          - ``"status"``          : ``STATUS_ANSWERED``, ``STATUS_REFUSED_WEAK_CONTEXT``,
                                    or ``STATUS_REFUSED_NO_CHUNKS``
          - ``"chunks_retrieved"``: number of chunks returned by retrieval

    Example::

        history = ConversationHistory(max_turns=3)
        result = conversational_answer(
            "What about the video requirement?",
            history,
            retrieve_fn=lambda q: retrieve(q, corpus, embed_fn, k=4),
            generate_fn=lambda q, chunks: {"answer": "...", "sources": []},
            rewrite_fn=lambda h, q: rewrite_followup(h, q, client, model),
        )
        print(result["rewritten_query"])
        print(result["answer"])
    """
    cfg = guardrail_config or RetrievalStrengthConfig()

    # ── Step 1: rewrite the follow-up ─────────────────────────────────────
    history_msgs = history.trimmed_messages()
    if rewrite_fn is not None:
        standalone_query = rewrite_fn(history_msgs, question)
    else:
        standalone_query = question

    # ── Step 2: retrieve with the rewritten query ──────────────────────────
    chunks = retrieve_fn(standalone_query)

    # ── Step 3: guardrail check ────────────────────────────────────────────
    if not chunks:
        answer = REFUSAL_MESSAGE_NO_CONTEXT
        status = STATUS_REFUSED_NO_CHUNKS
    elif not retrieval_is_strong(chunks, cfg):
        answer = REFUSAL_MESSAGE_NO_CONTEXT
        status = STATUS_REFUSED_WEAK_CONTEXT
    else:
        # ── Step 4: generate answer from original question + context ───────
        generated = generate_fn(question, chunks)
        answer = generated.get("answer", "")
        status = STATUS_ANSWERED

    # ── Step 5: update conversation history ───────────────────────────────
    if update_history:
        history.add_user(question)
        history.add_assistant(answer)

    sources = [c.get("metadata", {}) for c in chunks] if chunks else []

    return {
        "answer": answer,
        "rewritten_query": standalone_query,
        "sources": sources,
        "status": status,
        "chunks_retrieved": len(chunks),
    }


# ── Display helper ────────────────────────────────────────────────────────

def print_conversation_turn(
    turn_number: int,
    question: str,
    result: Dict[str, Any],
) -> None:
    """Print a single conversational RAG turn to stdout.

    Args:
        turn_number: 1-based turn counter for display.
        question: The original user question.
        result: The dict returned by :func:`conversational_answer`.
    """
    print(f"\n{'=' * 70}")
    print(f"TURN {turn_number}")
    print("=" * 70)
    print(f"  User question   : {question}")
    print(f"  Rewritten query : {result['rewritten_query']}")
    print(f"  Status          : {result['status']}")
    print(f"  Chunks retrieved: {result['chunks_retrieved']}")
    print(f"  Answer          : {result['answer'][:300]}")
    if result["sources"]:
        print("  Sources         :")
        for src in result["sources"]:
            print(f"    - {src.get('source', 'unknown')}#{src.get('chunk_index', 'n/a')}")
    print("=" * 70)
