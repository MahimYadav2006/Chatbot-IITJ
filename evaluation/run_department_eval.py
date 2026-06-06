#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.pipeline import (  # noqa: E402
    default_output_paths,
    evaluate_question_set,
    generate_question_set,
    load_dataset,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate department-specific chatbot questions using the configured LLM provider."
    )
    parser.add_argument(
        "--dept",
        default="computer_science_engineering",
        help="Department code or alias. Example: computer_science_engineering or cse",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the configured provider.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=24,
        help="How many evaluation questions to generate.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "evaluate"),
        default="all",
        help="Run the full pipeline, only generate questions, or only evaluate an existing dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional path for the generated or existing dataset JSON.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help="Optional path for evaluation results JSON.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional path for the markdown report.",
    )
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=10000,
        help="Maximum grounding bundle size for question generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_dataset, default_results, default_report = default_output_paths(args.dept)
    dataset_path = args.dataset_path or default_dataset
    results_path = args.results_path or default_results
    report_path = args.report_path or default_report

    if args.stage in {"all", "generate"}:
        logger.info("Generating question set for %s...", args.dept)
        generate_question_set(
            args.dept,
            question_count=args.question_count,
            model=args.model,
            output_path=dataset_path,
            max_source_chars=args.max_source_chars,
        )

    if args.stage in {"all", "evaluate"}:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. Run with --stage generate or --stage all first."
            )
        logger.info("Evaluating chatbot answers for %s...", args.dept)
        dataset = load_dataset(dataset_path)
        evaluate_question_set(
            dataset,
            dept_code=args.dept,
            model=args.model,
            results_path=results_path,
            report_path=report_path,
        )

    logger.info("Finished. Dataset: %s", dataset_path)
    if args.stage in {"all", "evaluate"}:
        logger.info("Results: %s", results_path)
        logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
