import os
import logging

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError


# Load environment variables from .env
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def main():
    # Read configuration from environment variables
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL")

    # Check if required environment variables exist
    if not base_url or not api_key or not model:
        print("Configuration error: Check your .env file.")
        print("Required variables:")
        print("- OPENAI_BASE_URL")
        print("- OPENAI_API_KEY")
        print("- CHAT_MODEL")
        return

    # Create OpenAI-compatible client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Messages sent to the model
    messages = [
        {
            "role": "system",
            "content": "You are a concise HR assistant."
        },
        {
            "role": "user",
            "content": "Say hello and explain what an HR assistant can help with in one sentence."
        }
    ]

    # Log outgoing request
    logging.info("REQUEST: %s", messages)
    logging.info("MODEL: %s", model)

    try:
        # Send chat completion request
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        # Extract model response
        answer = response.choices[0].message.content

        # Print response
        print("\nMODEL RESPONSE:")
        print(answer)

        # Log response
        logging.info("RESPONSE: %s", answer)

        # Log token usage if available
        if response.usage:
            logging.info("USAGE: %s", response.usage)

    except AuthenticationError:
        print(
            "\nAuthentication failed (401).\n"
            "Please check OPENAI_API_KEY in your .env file."
        )

    except RateLimitError:
        print(
            "\nRate limit or quota exceeded (429).\n"
            "Please wait and retry later."
        )

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()