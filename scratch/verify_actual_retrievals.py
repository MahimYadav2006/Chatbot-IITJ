"""Script to verify actual retrieval results from the ingested academics graph."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_section_retriever

def main():
    print("Loading academics retriever...")
    retriever = get_section_retriever("academics")
    if not retriever:
        print("❌ Failed to load academics retriever!")
        return

    queries = [
        "List all members of DPGC Committee of Chemical Engineering",
        "Who is DPGC Chairperson of chemical engineering?",
        "Who is DUGC Chairperson of chemical engineering?",
        "Who is the faculty advisor of B.Tech. in Civil Engineering?",
        "Who is the programme coordinator for BTP/MTP?",
        "What is the fee structure for B.Tech students admitted in 2025?",
        "What is the fee structure of PhD program for 2025 entry?",
        "Eligibility, details and procedure of BS honors",
        "Contingencies and Travel Supports policy",
        "Policy regarding conversion of II Grades"
    ]

    print("\n--- RUNNING QUERIES ---\n")
    for q in queries:
        print(f"Query: '{q}'")
        context = retriever.get_deterministic_context(q)
        if context:
            print("Answer (Deterministic Context):")
            print(context)
        else:
            print("No deterministic context found. Falling back to semantic/chunk search...")
            bundle = retriever.retrieve_bundle(q)
            chunks = bundle.context if hasattr(bundle, 'context') else str(bundle)
            print("Retrieved Chunks Context Snippet:")
            print("\n".join(chunks.split("\n")[:10]) + "\n...")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
