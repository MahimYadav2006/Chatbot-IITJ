#!/usr/bin/env python3
"""
GraphRAG Ingestion Pipeline for IIT Jammu EE Department.

Runs the full pipeline:
    1. Parse markdown files → extract entities & relationships
    2. Build NetworkX knowledge graph
    3. Run Louvain community detection
    4. Generate embeddings (chunks + entities + community summaries)
    5. Generate community summaries via LLM
    6. Persist everything to data/ directory

Usage:
    python ingest.py                  # Full pipeline
    python ingest.py --skip-summaries # Skip LLM summarization
"""

import os
import sys
import time
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Ingestion Pipeline")
    parser.add_argument("--skip-summaries", action="store_true",
                        help="Skip LLM-based community summarization")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Output directory for persisted data")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    total_start = time.time()

    # ── Step 1: Build Knowledge Graph ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Building Knowledge Graph")
    logger.info("=" * 60)

    from graphrag.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder()
    graph = builder.build()
    builder.save(args.data_dir)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_chunks = len(builder.chunks)
    logger.info(f"✅ Knowledge Graph: {n_nodes} nodes, {n_edges} edges, {n_chunks} text chunks")

    # ── Step 2: Community Detection ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Community Detection (Louvain)")
    logger.info("=" * 60)

    from graphrag.community import (
        detect_communities, build_community_reports,
        summarize_communities, save_communities
    )

    partition = detect_communities(graph, resolution=1.0)
    reports = build_community_reports(graph, partition)
    logger.info(f"✅ Detected {len(reports)} communities")

    # ── Step 3: Community Summarization ───────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Community Summarization")
    logger.info("=" * 60)

    if args.skip_summaries:
        logger.info("Skipping LLM summarization (--skip-summaries flag)")
        reports = summarize_communities(reports, llm_fn=None)
    else:
        from graphrag.llm import OllamaLLM
        llm = OllamaLLM()
        logger.info(f"Using Ollama LLM ({llm.model}) for summarization...")
        reports = summarize_communities(reports, llm_fn=llm)

    save_communities(reports, partition, args.data_dir)
    logger.info(f"✅ Community reports saved ({len(reports)} communities)")

    # ── Step 4: Generate Embeddings ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Generating Embeddings & Building FAISS Index")
    logger.info("=" * 60)

    from graphrag.embeddings import EmbeddingEngine, create_entity_descriptions

    engine = EmbeddingEngine()

    # Prepare chunks for embedding
    import json
    chunks_path = os.path.join(args.data_dir, "chunks.json")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    # Create entity descriptions
    entity_descriptions = create_entity_descriptions(graph)
    logger.info(f"Created {len(entity_descriptions)} entity descriptions")

    # Prepare community summaries for embedding
    community_items = []
    for report in reports:
        community_items.append({
            "id": report["id"],
            "text": report.get("summary", report["text"]),
            "metadata": {
                "community_id": report["community_id"],
                "size": report["size"]
            }
        })

    # Build FAISS index
    engine.build_index(chunks, entity_descriptions, community_items)
    engine.save(args.data_dir)

    logger.info(f"✅ FAISS index built: {engine.index.ntotal} vectors")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 INGESTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Knowledge Graph: {n_nodes} nodes, {n_edges} edges")
    logger.info(f"  Text Chunks:     {n_chunks}")
    logger.info(f"  Communities:     {len(reports)}")
    logger.info(f"  FAISS Vectors:   {engine.index.ntotal}")
    logger.info(f"  Entity Descs:    {len(entity_descriptions)}")
    logger.info(f"  Time Elapsed:    {elapsed:.1f}s")
    logger.info(f"  Output Dir:      {args.data_dir}")
    logger.info("")
    logger.info("You can now start the chatbot with: python app.py")


if __name__ == "__main__":
    main()
