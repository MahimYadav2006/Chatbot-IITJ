#!/usr/bin/env python3
"""
GraphRAG Ingestion Pipeline for IIT Jammu Departments.

Runs the full pipeline for a specific department or all departments:
    1. Parse markdown files → extract entities & relationships
    2. Build NetworkX knowledge graph
    3. Run Louvain community detection
    4. Generate embeddings (chunks + entities + community summaries)
    5. Generate community summaries via LLM
    6. Persist everything to data/{dept_code}/ directory

Usage:
    python ingest.py --dept ee         # Ingest Electrical Engineering
    python ingest.py --dept cse        # Ingest Computer Science
    python ingest.py --all             # Ingest all 11+ departments
    python ingest.py --skip-summaries  # Ingest default department skipping LLM community summaries
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


def ingest_department(dept_code: str, skip_summaries: bool = False):
    from departments import get_markdown_dir, get_data_dir, get_department, resolve_department_code
    
    canonical_code = resolve_department_code(dept_code)
    dept_config = get_department(canonical_code)
    markdown_dir = get_markdown_dir(canonical_code)
    data_dir = get_data_dir(canonical_code)
    
    os.makedirs(data_dir, exist_ok=True)
    start_time = time.time()
    
    # ── Step 1: Build Knowledge Graph ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 1: Building Knowledge Graph for {dept_config['full_name']}")
    logger.info("=" * 60)

    from graphrag.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(dept_code=canonical_code)
    graph = builder.build()
    builder.save(data_dir)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_chunks = len(builder.chunks)
    logger.info(f"✅ Knowledge Graph: {n_nodes} nodes, {n_edges} edges, {n_chunks} text chunks")

    # ── Step 2: Community Detection ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 2: Community Detection (Louvain) for {canonical_code.upper()}")
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
    logger.info(f"STEP 3: Community Summarization for {canonical_code.upper()}")
    logger.info("=" * 60)

    if skip_summaries:
        logger.info("Skipping LLM summarization (--skip-summaries flag)")
        reports = summarize_communities(reports, llm_fn=None)
    else:
        from graphrag.llm import OllamaLLM
        llm = OllamaLLM()
        logger.info(f"Using Ollama LLM ({llm.model}) for summarization...")
        reports = summarize_communities(reports, llm_fn=llm)

    save_communities(reports, partition, data_dir)
    logger.info(f"✅ Community reports saved ({len(reports)} communities)")

    # ── Step 4: Generate Embeddings ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 4: Generating Embeddings & FAISS Index for {canonical_code.upper()}")
    logger.info("=" * 60)

    from graphrag.embeddings import EmbeddingEngine, create_entity_descriptions

    engine = EmbeddingEngine()

    # Prepare chunks for embedding
    import json
    chunks_path = os.path.join(data_dir, "chunks.json")
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
    engine.build_index(chunks, entity_descriptions, community_items, dept_code=canonical_code)
    engine.save(data_dir)

    logger.info(f"✅ FAISS index built: {engine.index.ntotal} vectors")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🎉 INGESTION COMPLETE FOR {canonical_code.upper()}!")
    logger.info("=" * 60)
    logger.info(f"  Knowledge Graph: {n_nodes} nodes, {n_edges} edges")
    logger.info(f"  Text Chunks:     {n_chunks}")
    logger.info(f"  Communities:     {len(reports)}")
    logger.info(f"  FAISS Vectors:   {engine.index.ntotal}")
    logger.info(f"  Entity Descs:    {len(entity_descriptions)}")
    logger.info(f"  Time Elapsed:    {elapsed:.1f}s")
    logger.info(f"  Output Dir:      {data_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="IIT Jammu Multi-Department GraphRAG Ingestion Pipeline")
    parser.add_argument("--dept", default="ee",
                        help="Department code to ingest (e.g. ee, cse, computer_science_engineering, bsbe)")
    parser.add_argument("--all", action="store_true",
                        help="Ingest all registered academic departments")
    parser.add_argument("--skip-summaries", action="store_true",
                        help="Skip LLM-based community summarization (faster, cheaper)")
    args = parser.parse_args()

    from departments import DEPARTMENTS, resolve_department_code

    if args.all:
        logger.info(f"Ingesting ALL {len(DEPARTMENTS)} academic departments...")
        for dept in DEPARTMENTS:
            logger.info(f"\n\n>>> Starting Ingestion for academic department: {dept.upper()} <<<")
            try:
                ingest_department(dept, skip_summaries=args.skip_summaries)
            except Exception as e:
                logger.error(f"❌ Failed to ingest academic department {dept.upper()}: {e}")
    else:
        dept = args.dept.lower()
        try:
            canonical_dept = resolve_department_code(dept)
        except KeyError:
            logger.error(f"❌ Unknown department code: {dept}. Must be one of: {list(DEPARTMENTS.keys())}")
            sys.exit(1)
        ingest_department(canonical_dept, skip_summaries=args.skip_summaries)


if __name__ == "__main__":
    main()
