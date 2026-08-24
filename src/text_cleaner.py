import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean raw extracted document text and prepare it
    for retrieval and embedding.
    """

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove repeated header
    text = re.sub(
        r"HR ASSIST EMPLOYEE POLICY",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove page footers
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Collapse multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_corpus(documents):
    """
    Apply the same cleaning pipeline
    to every document in the corpus.
    """

    cleaned_documents = []

    for document in documents:

        before_text = document["text"]

        after_text = clean_text(before_text)

        cleaned_document = {
            "source": document["source"],
            "text": after_text
        }

        cleaned_documents.append(cleaned_document)

    return cleaned_documents


def show_before_after(documents, cleaned_documents):

    print("=" * 70)
    print("TEXT EXTRACTION & CLEANING PIPELINE")
    print("=" * 70)

    for before_doc, after_doc in zip(
        documents,
        cleaned_documents
    ):

        print("\nSOURCE:", before_doc["source"])

        print("\nBEFORE CLEANING:")
        print("-" * 40)
        print(before_doc["text"][:500])

        print("\nAFTER CLEANING:")
        print("-" * 40)
        print(after_doc["text"][:500])

        print("\nCHARACTER COUNT:")
        print(
            f"{len(before_doc['text'])} -> "
            f"{len(after_doc['text'])}"
        )

        print("\n" + "=" * 70)


sample_documents = [
    {
        "source": "employee_policy.txt",
        "text": """
HR ASSIST EMPLOYEE POLICY

Page 1 of 5


Employee Leave Policy

Employees are entitled to    annual leave,
sick leave,    and casual leave.


HR ASSIST EMPLOYEE POLICY
Page 2 of 5

Leave requests must be submitted
through the employee portal.
"""
    },
    {
        "source": "attendance_policy.txt",
        "text": """
HR ASSIST EMPLOYEE POLICY

Page 1 of 3


Attendance Policy


Employees must report to work
on time.     Repeated lateness
may result in disciplinary action.


Page 2 of 3

HR ASSIST EMPLOYEE POLICY

Employees should notify their manager
if they are unable to attend work.
"""
    },
    {
        "source": "remote_work_policy.txt",
        "text": """
HR ASSIST EMPLOYEE POLICY

Page 1 of 2


Remote Work Policy


Employees may work remotely
when approved by their manager.



    Communication must remain clear
    and professional.


Page 2 of 2
HR ASSIST EMPLOYEE POLICY
"""
    }
]


def main():

    print("\nCleaning documents...\n")

    cleaned_documents = clean_corpus(
        sample_documents
    )

    show_before_after(
        sample_documents,
        cleaned_documents
    )


if __name__ == "__main__":
    main()