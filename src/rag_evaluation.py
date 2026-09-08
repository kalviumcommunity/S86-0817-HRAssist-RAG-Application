import json
from src.grounded_generation import (
    retrieve,
    generate_grounded_answer
)


# ============================================================
# TEST SET
# ============================================================

test_set = [
    {
        "question": "What are the password reset steps?",
        "expected_points": [
            "password reset",
            "registered email"
        ],
        "expected_sources": {
            "account-guide.md"
        }
    },

    {
        "question": "What evidence is required for project submission?",
        "expected_points": [
            "PR link",
            "sample output",
            "video explanation"
        ],
        "expected_sources": {
            "submission-rubric.md"
        }
    },

    {
        "question": "What should the system do when information is missing?",
        "expected_points": [
            "not enough information",
            "provided context"
        ],
        "expected_sources": {
            "guardrails.md"
        }
    }
]


# ============================================================
# CORRECTNESS SCORING
# ============================================================

def score_correctness(answer, expected_points):

    answer_lower = answer.lower()

    matched_points = []

    for point in expected_points:

        if point.lower() in answer_lower:
            matched_points.append(point)

    if not expected_points:
        return 0

    score = len(matched_points) / len(expected_points)

    return round(score, 2)


# ============================================================
# GROUNDING SCORING
# ============================================================

def score_grounding(answer, retrieved_chunks):

    if not retrieved_chunks:
        return 0

    answer_lower = answer.lower()

    supported_claims = 0
    total_claims = 0

    # Split answer into simple sentences
    sentences = [
        sentence.strip()
        for sentence in answer.split(".")
        if sentence.strip()
    ]

    context = " ".join(
        chunk["text"].lower()
        for chunk in retrieved_chunks
    )

    for sentence in sentences:

        total_claims += 1

        words = [
            word.strip(".,!?;:")
            for word in sentence.lower().split()
            if len(word) > 3
        ]

        matching_words = [
            word
            for word in words
            if word in context
        ]

        # Simple heuristic:
        # if at least 30% of meaningful words
        # appear in retrieved context,
        # consider the sentence supported.
        if words and len(matching_words) / len(words) >= 0.3:
            supported_claims += 1

    if total_claims == 0:
        return 0

    return round(
        supported_claims / total_claims,
        2
    )


# ============================================================
# CITATION ACCURACY
# ============================================================

def check_citation_accuracy(
    citations,
    expected_sources
):

    if not citations:
        return 0

    cited_sources = {
        citation["source"]
        for citation in citations
    }

    matches = cited_sources.intersection(
        expected_sources
    )

    if not expected_sources:
        return 0

    return round(
        len(matches) / len(expected_sources),
        2
    )


# ============================================================
# RUN ONE TEST
# ============================================================

def evaluate_question(example):

    question = example["question"]

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # Retrieve
    retrieved_chunks = retrieve(
        question,
        k=4
    )

    # Generate grounded answer
    result = generate_grounded_answer(
        question,
        retrieved_chunks
    )

    answer = result["answer"]

    # Scores
    correctness = score_correctness(
        answer,
        example["expected_points"]
    )

    grounding = score_grounding(
        answer,
        retrieved_chunks
    )

    citation_accuracy = check_citation_accuracy(
        result["sources"],
        example["expected_sources"]
    )

    print("\nANSWER:")
    print(answer)

    print("\nEXPECTED POINTS:")
    print(example["expected_points"])

    print("\nRETRIEVED SOURCES:")

    for source in result["sources"]:

        print(
            f"- {source['source']} "
            f"| {source['section']}"
        )

    print("\nSCORES:")
    print(
        "Correctness:",
        correctness
    )

    print(
        "Grounding:",
        grounding
    )

    print(
        "Citation Accuracy:",
        citation_accuracy
    )

    return {
        "question": question,
        "answer": answer,
        "expected_points": example["expected_points"],
        "expected_sources": list(
            example["expected_sources"]
        ),
        "citations": result["sources"],
        "correctness": correctness,
        "grounding": grounding,
        "citation_accuracy": citation_accuracy
    }


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def identify_failure(result):

    failures = []

    if result["correctness"] < 1:
        failures.append(
            "Expected answer points were missing."
        )

    if result["grounding"] < 1:
        failures.append(
            "Some answer claims may not be fully "
            "supported by retrieved context."
        )

    if result["citation_accuracy"] < 1:
        failures.append(
            "Citations did not fully match "
            "the expected sources."
        )

    if not failures:
        failures.append(
            "No major failure detected."
        )

    return failures


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("RAG EVALUATION & ANSWER QUALITY SCORING")
    print("=" * 70)

    results = []

    for example in test_set:

        result = evaluate_question(
            example
        )

        result["failure_analysis"] = (
            identify_failure(result)
        )

        results.append(result)


    # ========================================================
    # SUMMARY
    # ========================================================

    number_of_questions = len(results)

    average_correctness = sum(
        result["correctness"]
        for result in results
    ) / number_of_questions

    average_grounding = sum(
        result["grounding"]
        for result in results
    ) / number_of_questions

    average_citation_accuracy = sum(
        result["citation_accuracy"]
        for result in results
    ) / number_of_questions


    overall_score = (
        average_correctness
        + average_grounding
        + average_citation_accuracy
    ) / 3


    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        "Questions evaluated:",
        number_of_questions
    )

    print(
        "Average correctness:",
        round(average_correctness, 2)
    )

    print(
        "Average grounding:",
        round(average_grounding, 2)
    )

    print(
        "Average citation accuracy:",
        round(average_citation_accuracy, 2)
    )

    print(
        "Overall RAG quality:",
        round(overall_score, 2)
    )


    # ========================================================
    # FAILURE REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    for result in results:

        if (
            result["correctness"] < 1
            or result["grounding"] < 1
            or result["citation_accuracy"] < 1
        ):

            print(
                "\nFailed/partial question:",
                result["question"]
            )

            print(
                "Correctness:",
                result["correctness"]
            )

            print(
                "Grounding:",
                result["grounding"]
            )

            print(
                "Citation Accuracy:",
                result["citation_accuracy"]
            )

            for failure in result[
                "failure_analysis"
            ]:

                print(
                    "-",
                    failure
                )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    evaluation_report = {
        "questions": number_of_questions,
        "average_correctness": round(
            average_correctness,
            2
        ),
        "average_grounding": round(
            average_grounding,
            2
        ),
        "average_citation_accuracy": round(
            average_citation_accuracy,
            2
        ),
        "overall_quality": round(
            overall_score,
            2
        ),
        "results": results
    }


    with open(
        "rag_evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=2,
            ensure_ascii=False
        )


    print("\nEvaluation report saved to:")
    print("rag_evaluation_results.json")


if __name__ == "__main__":
    main()