"""Reusable prompt templates for HRAssist responses."""

ANSWER_TEMPLATE = (
    "You are a support assistant. Answer ONLY from the context.\n"
    "If the answer isn't there, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def render_template(template: str, **values) -> str:
    """Fill a prompt template with runtime values."""
    return template.format(**values)
