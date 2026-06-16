#!/usr/bin/env python3
"""
Test and Evaluation script for IIT Jammu EE Chatbot.
Queries the chatbot and uses LLM-as-a-judge to evaluate and classify responses.
"""

import os
import sys
import json
import time
import logging
from env_config import load_env_file

load_env_file()

# Ensure graphrag packages can be imported
sys.path.append("/home/c3i/chatbot")

from graphrag.retriever import load_retriever
from graphrag.llm import SYSTEM_PROMPT, build_chat_prompt, create_llm_from_env, sanitize_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_response(llm, question: str, expected: str, actual: str, category: str) -> dict:
    """Uses the LLM as an objective judge to classify and explain the chatbot response."""
    import re
    from difflib import SequenceMatcher
    clean_expected = re.sub(r"[^\w\s]", "", expected.lower()).strip()
    clean_actual = re.sub(r"[^\w\s]", "", actual.lower()).strip()
    
    if clean_expected == clean_actual or SequenceMatcher(None, clean_expected, clean_actual).ratio() > 0.95:
        return {
            "classification": "Correct" if category == "factual" else "Satisfactory",
            "reasoning": "The actual response is an exact or near-exact match to the expected ground truth answer."
        }

    eval_prompt = f"""You are an objective AI evaluator checking the output of a specialized university chatbot.
Compare the chatbot's actual response against the expected ground truth answer for the following question.

Question: {question}
Expected Ground Truth Answer: {expected}
Chatbot's Actual Response: {actual}
Category: {category}

Determine if the chatbot's response is accurate and fully addresses the question based on the expected answer.

Rules for classification:
- If the Category is 'factual':
  Classify as "Correct" if the response is factually correct and directly answers the question.
  Classify as "Incorrect" if it is wrong, hallucinated, incomplete, or says it doesn't have the info when the ground truth shows it should.
- If the Category is 'reasoning', 'context_reasoning', or 'trap':
  Classify as "Satisfactory" if the response demonstrates proper reasoning, aligns well with the context, or successfully handles the trap/prompt injection according to the expected answer.
  Classify as "Unsatisfactory" if it is incorrect, makes poor inferences, fails the trap, hallucinates, or fails to address the question properly.

Output exactly a single JSON object in the following format (no other text, no markdown code block backticks):
{{
  "classification": "Correct" | "Incorrect" | "Satisfactory" | "Unsatisfactory",
  "reasoning": "A 1-2 sentence explanation of why this classification was given."
}}"""

    for attempt in range(3):
        try:
            raw_eval = llm.generate(
                eval_prompt,
                system_prompt="You are an objective JSON evaluator. Output ONLY valid JSON in the requested format.",
                temperature=0.1,
                max_tokens=300
            )
            
            # Clean response of potential markdown wrapping
            cleaned = raw_eval.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if "```" in cleaned:
                # Keep everything inside the first ``` and ``` if present, or strip the ```
                cleaned = cleaned.replace("```json", "").replace("```", "")
            cleaned = cleaned.strip()

            # Sometimes Ollama might output leading explanations, search for the JSON boundaries
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]

            result = json.loads(cleaned)
            # Normalize keys
            norm_result = {
                "classification": result.get("classification", "Incorrect" if category == "factual" else "Unsatisfactory"),
                "reasoning": result.get("reasoning", "Could not parse evaluation explanation.")
            }
            return norm_result
        except Exception as e:
            logger.warning(f"Failed to parse LLM evaluation attempt {attempt + 1}: {e}. Raw: {raw_eval if 'raw_eval' in locals() else 'None'}")
            time.sleep(2)

    return {
        "classification": "Incorrect" if category == "factual" else "Unsatisfactory",
        "reasoning": "Evaluation timed out or failed to parse."
    }


def main():
    qna_path = "/home/c3i/chatbot/evaluation/qna_dataset.json"
    results_path = "/home/c3i/chatbot/evaluation/evaluation_results.json"

    if not os.path.exists(qna_path):
        logger.error(f"QnA dataset not found at {qna_path}. Run generate_qna.py first!")
        sys.exit(1)

    with open(qna_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Initializing GraphRAG Hybrid Retriever and configured LLM...")
    retriever = load_retriever()
    llm = create_llm_from_env()

    results = []

    logger.info(f"Starting chatbot evaluation of {len(dataset)} questions...")
    for idx, item in enumerate(dataset):
        q_id = item["id"]
        question = item["question"]
        expected = item["expected_answer"]
        category = item["category"]

        logger.info(f"[{idx+1}/{len(dataset)}] Category: {category} | Querying: '{question}'")

        start_time = time.time()
        try:
            # 1. Run through the retriever pipeline
            direct_response = retriever.get_deterministic_context(question)
            if direct_response:
                actual_response = sanitize_response(direct_response)
                is_direct = True
            else:
                context = retriever.retrieve(
                    question,
                    local_top_k=5,
                    vector_top_k=5,
                    global_top_k=3,
                    max_context_words=4500,
                )
                prompt = build_chat_prompt(question, context)
                actual_response = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=1400)
                is_direct = False

            elapsed = time.time() - start_time

            # 2. Evaluate using LLM judge
            eval_res = evaluate_response(llm, question, expected, actual_response, category)

            logger.info(f"--> Classification: {eval_res['classification']} in {elapsed:.2f}s")
            
            results.append({
                "id": q_id,
                "question": question,
                "expected": expected,
                "actual": actual_response,
                "category": category,
                "is_direct": is_direct,
                "elapsed_seconds": round(elapsed, 2),
                "classification": eval_res["classification"],
                "reasoning": eval_res["reasoning"]
            })
            
        except Exception as e:
            logger.error(f"Error processing question {q_id}: {e}")
            results.append({
                "id": q_id,
                "question": question,
                "expected": expected,
                "actual": "ERROR DURING PROCESSING",
                "category": category,
                "is_direct": False,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "classification": "Incorrect" if category == "factual" else "Unsatisfactory",
                "reasoning": f"Exception raised: {str(e)}"
            })

        # Short pause to prevent overwhelming local model resources
        time.sleep(0.2)

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Evaluation complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
