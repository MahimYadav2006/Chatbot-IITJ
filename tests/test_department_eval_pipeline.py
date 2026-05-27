import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.pipeline import (
    ALLOWED_CATEGORIES,
    allocate_category_counts,
    extract_json_payload,
    summarize_results,
)


def test_allocate_category_counts_covers_all_categories():
    counts = allocate_category_counts(24)
    assert sum(counts.values()) == 24
    assert set(counts) == set(ALLOWED_CATEGORIES)
    assert all(value >= 1 for value in counts.values())


def test_extract_json_payload_handles_markdown_fences():
    raw = """```json
    [
      {"category": "factual", "question": "Q?", "expected_answer": "A"}
    ]
    ```"""
    payload = extract_json_payload(raw)
    assert isinstance(payload, list)
    assert payload[0]["category"] == "factual"


def test_summarize_results_uses_half_credit_for_partial():
    summary = summarize_results(
        [
            {
                "category": "factual",
                "evaluation": {"verdict": "pass", "score": 5},
            },
            {
                "category": "reasoning",
                "evaluation": {"verdict": "partial", "score": 3},
            },
            {
                "category": "reasoning",
                "evaluation": {"verdict": "fail", "score": 1},
            },
        ]
    )
    assert summary["total_questions"] == 3
    assert summary["verdict_counts"] == {"pass": 1, "partial": 1, "fail": 1}
    assert summary["weighted_pass_rate"] == 50.0
    assert summary["category_summary"]["reasoning"]["weighted_pass_rate"] == 25.0
