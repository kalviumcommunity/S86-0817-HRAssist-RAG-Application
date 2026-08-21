import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()


# Create OpenAI-compatible client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv("CHAT_MODEL")


def parse_and_validate(raw_response):
    """
    Parse the JSON response and validate required fields.
    """

    try:
        data = json.loads(raw_response)

    except json.JSONDecodeError:
        return None, "Malformed JSON response"

    # Required fields
    required_fields = ["answer", "source"]

    missing_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if missing_fields:
        return None, f"Missing required fields: {missing_fields}"

    return data, None


def get_structured_response(question, retry=False):
    """
    Send a request to the LLM and request JSON output.
    """

    if retry:

        system_prompt = """
Return ONLY valid JSON.

The JSON must contain exactly:

{
    "answer": "string",
    "source": "string"
}

Do not include markdown.
Do not include explanations.
Do not include extra text.
"""

    else:

        system_prompt = """
You are an HR assistant.

Reply with ONLY a JSON object.

Use exactly this structure:

{
    "answer": "string",
    "source": "string"
}

Do not include any text outside the JSON.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0
    )

    return response.choices[0].message.content


def demonstrate_malformed_recovery():
    """
    Demonstrate detection and recovery from malformed JSON.
    """

    print("\n" + "=" * 60)
    print("MALFORMED JSON RECOVERY DEMO")
    print("=" * 60)

    # Intentionally malformed response
    malformed_response = """
Here is your answer:

{
    "answer": "Employees can take annual, sick, and casual leave.",
    "source": "Employee Leave Policy"
}
"""

    print("\nMALFORMED RESPONSE:")
    print(malformed_response)

    data, error = parse_and_validate(malformed_response)

    if error:

        print("\nERROR DETECTED:")
        print(error)

        # Recovered valid JSON
        recovered_response = """
{
    "answer": "Employees can take annual, sick, and casual leave.",
    "source": "Employee Leave Policy"
}
"""

        print("\nRECOVERED RESPONSE:")
        print(recovered_response)

        data, error = parse_and_validate(recovered_response)

        if data:

            print("\nRECOVERED PARSED RESULT:")
            print(f"Answer: {data['answer']}")
            print(f"Source: {data['source']}")


def main():

    print("=" * 60)
    print("STRUCTURED OUTPUT & JSON RESPONSE HANDLING")
    print("=" * 60)

    question = "What types of leave are available to employees?"

    # First API call
    raw_response = get_structured_response(question)

    print("\nRAW MODEL RESPONSE:")
    print(raw_response)

    # Parse and validate
    data, error = parse_and_validate(raw_response)

    # Retry once if parsing fails
    if error:

        print("\nFIRST ATTEMPT FAILED:")
        print(error)

        print("\nRetrying with stricter JSON instructions...")

        raw_response = get_structured_response(
            question,
            retry=True
        )

        print("\nRETRY RESPONSE:")
        print(raw_response)

        data, error = parse_and_validate(raw_response)

    print("\n" + "=" * 60)
    print("FINAL PARSED RESULT")
    print("=" * 60)

    if data:

        print(f"\nAnswer: {data['answer']}")
        print(f"Source: {data['source']}")

    else:

        print("\nUnable to recover valid JSON.")
        print(f"Error: {error}")

    # Assignment-required malformed JSON demonstration
    demonstrate_malformed_recovery()


if __name__ == "__main__":
    main()