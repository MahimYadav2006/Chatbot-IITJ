#!/usr/bin/env python3
"""
Orchestrator script to run the full QnA generation and evaluation suite,
and generate a beautiful markdown report.
"""

import os
import sys
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, desc):
    logger.info(f"Running: {desc}...")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"❌ Failed: {desc}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        sys.exit(1)
    logger.info(f"✅ Success: {desc}")
    print(res.stdout)


def main():
    eval_dir = "/home/c3i/chatbot/evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Generate QnA pairs
    run_command(f"{sys.executable} {eval_dir}/generate_qna.py", "QnA Dataset Generation")

    # 2. Run test evaluation
    run_command(f"{sys.executable} {eval_dir}/test_evaluation.py", "Chatbot Test Evaluation")

    # 3. Generate Report
    results_path = f"{eval_dir}/evaluation_results.json"
    if not os.path.exists(results_path):
        logger.error(f"Evaluation results not found at {results_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Calculate metrics
    total = len(results)
    category_stats = {}

    for item in results:
        cat = item["category"]
        cls = item["classification"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "pass": 0, "fail": 0}
        
        category_stats[cat]["total"] += 1
        if cls in ("Correct", "Satisfactory"):
            category_stats[cat]["pass"] += 1
        else:
            category_stats[cat]["fail"] += 1

    total_factual = category_stats.get("factual", {}).get("total", 0)
    pass_factual = category_stats.get("factual", {}).get("pass", 0)
    
    total_reasoning = category_stats.get("reasoning", {}).get("total", 0)
    pass_reasoning = category_stats.get("reasoning", {}).get("pass", 0)

    total_context = category_stats.get("context_reasoning", {}).get("total", 0)
    pass_context = category_stats.get("context_reasoning", {}).get("pass", 0)

    total_trap = category_stats.get("trap", {}).get("total", 0)
    pass_trap = category_stats.get("trap", {}).get("pass", 0)

    total_passed = pass_factual + pass_reasoning + pass_context + pass_trap
    overall_accuracy = (total_passed / total) * 100 if total > 0 else 0

    # Build markdown report
    report_lines = [
        "# IIT Jammu EE Chatbot Evaluation Report",
        "",
        "This report summarizes the performance of the Department of Electrical Engineering GraphRAG Chatbot at IIT Jammu. A comprehensive evaluation dataset covering almost all files (faculty roster, individual faculty profiles, PhD lists, patents, projects, startups, placements, curriculum, and research areas) was generated. The dataset includes factual, reasoning-based, context reasoning, and chatbot trap queries.",
        "",
        "## Overall Summary",
        "",
        f"- **Total Questions Evaluated**: {total}",
        f"- **Overall Performance Rate**: {overall_accuracy:.2f}% ({total_passed}/{total} Passed)",
        "",
        "### Performance by Category",
        "",
        "| Category | Total Questions | Passed (Correct / Satisfactory) | Failed (Incorrect / Unsatisfactory) | Pass Rate |",
        "| --- | --- | --- | --- | --- |",
    ]

    for cat_name, label in [
        ("factual", "Factual (Correct/Incorrect)"),
        ("reasoning", "Reasoning (Satisfactory/Unsatisfactory)"),
        ("context_reasoning", "Context Reasoning (Satisfactory/Unsatisfactory)"),
        ("trap", "Chatbot Trap Queries (Satisfactory/Unsatisfactory)"),
    ]:
        stats = category_stats.get(cat_name, {"total": 0, "pass": 0, "fail": 0})
        rate = (stats["pass"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        report_lines.append(f"| **{label}** | {stats['total']} | {stats['pass']} | {stats['fail']} | {rate:.2f}% |")

    report_lines.extend([
        "",
        "## Detailed Results",
        "",
        "Below is the complete evaluation breakdown for all QnA pairs, including the question, expected ground-truth answer, the actual chatbot response, classification, and evaluator's reasoning.",
        ""
    ])

    for item in results:
        icon = "✅" if item["classification"] in ("Correct", "Satisfactory") else "❌"
        direct_tag = " [Direct Graph Answer]" if item["is_direct"] else ""
        report_lines.extend([
            f"### Question {item['id']}: {item['question']}",
            "",
            f"- **Category**: `{item['category']}`",
            f"- **Evaluation**: {icon} **{item['classification']}**{direct_tag} (Response Time: {item['elapsed_seconds']}s)",
            f"- **Evaluator Explanation**: *{item['reasoning']}*",
            "",
            "<details>",
            "<summary>Show expected vs actual response</summary>",
            "",
            "**Expected Ground Truth Answer:**",
            f"> {item['expected']}",
            "",
            "**Chatbot's Actual Response:**",
            f"> {item['actual']}",
            "",
            "</details>",
            "",
            "---",
            ""
        ])

    report_path = f"{eval_dir}/evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"🎉 Evaluation report successfully written to {report_path}")


if __name__ == "__main__":
    main()
