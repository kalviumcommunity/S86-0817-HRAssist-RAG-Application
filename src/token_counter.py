from pathlib import Path
import tiktoken


# Use the cl100k_base tokenizer
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in a text."""
    return len(encoding.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int
):
    """
    Calculate estimated cost.

    These are example rates per 1,000 tokens.
    """

    INPUT_RATE = 0.0005
    OUTPUT_RATE = 0.0015

    input_cost = (input_tokens / 1000) * INPUT_RATE
    output_cost = (output_tokens / 1000) * OUTPUT_RATE

    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


def main():

    # ----------------------------------
    # Sample 1: Short Question
    # ----------------------------------

    short_question = "How many sick leaves do I have?"

    # ----------------------------------
    # Sample 2: Paragraph
    # ----------------------------------

    paragraph = """
Employees are entitled to annual leave, sick leave, and casual leave
according to company policy. Annual leave must be requested through
the HR system before the planned absence whenever possible.
"""

    # ----------------------------------
    # Sample 3: Full Document
    # ----------------------------------

    project_root = Path(__file__).resolve().parent.parent
    policy_path = project_root / "data" / "policy.txt"

    full_document = policy_path.read_text(
        encoding="utf-8"
    )

    # ----------------------------------
    # Additional samples
    # ----------------------------------

    samples = {
        "Short Question": short_question,
        "Paragraph": paragraph,
        "Full Document": full_document,
        "Long Word": "antidisestablishmentarianism",
        "Code": "for(int i = 0; i < n; i++) { System.out.println(i); }",
        "Telugu Text": "నాకు ఉద్యోగుల సెలవు విధానం గురించి సమాచారం కావాలి"
    }

    print("=" * 60)
    print("TOKEN COUNT REPORT")
    print("=" * 60)

    results = {}

    # ----------------------------------
    # Count tokens for every sample
    # ----------------------------------

    for name, text in samples.items():

        characters = len(text)
        words = len(text.split())
        tokens = count_tokens(text)

        results[name] = tokens

        print(f"\nSample: {name}")
        print(f"Characters: {characters}")
        print(f"Words: {words}")
        print(f"Tokens: {tokens}")

    # ----------------------------------
    # Cost Estimation
    # ----------------------------------

    print("\n" + "=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)

    input_tokens = results["Full Document"]

    output_text = """
Employees can request annual, sick, or casual leave according
to company policy. Contact HR for information about your leave balance.
"""

    output_tokens = count_tokens(output_text)

    input_cost, output_cost, total_cost = estimate_cost(
        input_tokens,
        output_tokens
    )

    print(f"\nInput Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}")

    print(f"\nInput Cost: ${input_cost:.8f}")
    print(f"Output Cost: ${output_cost:.8f}")
    print(f"Estimated Total Cost: ${total_cost:.8f}")

    # ----------------------------------
    # Observation
    # ----------------------------------

    print("\n" + "=" * 60)
    print("OBSERVATION")
    print("=" * 60)

    print("""
Text length and token count generally increase together,
but they are not exactly proportional.

Long words, programming code, and different languages
can produce different token counts.
""")


if __name__ == "__main__":
    main()