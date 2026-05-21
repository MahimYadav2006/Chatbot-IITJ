"""
Community Detection and Summarization for GraphRAG.

Implements:
    1. Louvain community detection on the entity graph
    2. Hierarchical community structure at multiple resolutions
    3. Community summarization via LLM
"""

import os
import json
import logging
from collections import defaultdict
from typing import List, Dict, Optional

import networkx as nx

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def detect_communities(graph: nx.DiGraph, resolution: float = 1.0) -> Dict[str, int]:
    """
    Run Louvain community detection on the entity graph.

    We convert to undirected and filter to only entity nodes
    (excluding TextChunk and Document nodes) for meaningful communities.

    Args:
        graph: NetworkX directed graph
        resolution: Louvain resolution parameter (higher = more communities)

    Returns:
        Dict mapping node_id -> community_id
    """
    import community as community_louvain

    # Build undirected subgraph with only entity nodes
    entity_labels = {"Faculty", "PhDStudent", "ResearchArea", "ResearchCategory",
                     "Project", "Patent", "Startup", "Lab", "Department",
                     "Programme", "FundingAgency", "Placement", "ExternalPerson"}

    entity_nodes = [n for n, d in graph.nodes(data=True)
                    if d.get("label") in entity_labels]

    if not entity_nodes:
        logger.warning("No entity nodes found for community detection.")
        return {}

    # Create undirected subgraph
    subgraph = graph.subgraph(entity_nodes).to_undirected()

    # Remove isolated nodes
    connected_nodes = [n for n in subgraph.nodes() if subgraph.degree(n) > 0]
    subgraph = subgraph.subgraph(connected_nodes)

    if len(subgraph) == 0:
        logger.warning("No connected entity nodes for community detection.")
        return {}

    # Run Louvain
    partition = community_louvain.best_partition(subgraph, resolution=resolution, random_state=42)

    # Count communities
    community_counts = defaultdict(int)
    for node, comm_id in partition.items():
        community_counts[comm_id] += 1

    logger.info(f"Detected {len(community_counts)} communities from {len(partition)} entities")
    for comm_id, count in sorted(community_counts.items()):
        logger.info(f"  Community {comm_id}: {count} entities")

    return partition


def build_community_reports(graph: nx.DiGraph, partition: Dict[str, int]) -> List[Dict]:
    """
    Build structured reports for each community.

    Each report contains the community's member entities, their types,
    key relationships, and a text representation for embedding/summarization.

    Returns:
        List of community report dicts
    """
    if not partition:
        return []

    # Group nodes by community
    communities = defaultdict(list)
    for node_id, comm_id in partition.items():
        communities[comm_id].append(node_id)

    reports = []

    for comm_id, members in sorted(communities.items()):
        # Categorize members by type
        members_by_type = defaultdict(list)
        for node_id in members:
            data = graph.nodes[node_id]
            label = data.get("label", "Unknown")
            members_by_type[label].append(node_id)

        # Get internal relationships
        internal_edges = []
        for u, v, edge_data in graph.edges(data=True):
            if u in partition and v in partition:
                if partition.get(u) == comm_id and partition.get(v) == comm_id:
                    internal_edges.append({
                        "source": u,
                        "target": v,
                        "type": edge_data.get("type", "RELATED")
                    })

        # Build text representation for embedding
        text_parts = []
        text_parts.append(f"Community {comm_id} in IIT Jammu EE Department:")

        for label, nodes in sorted(members_by_type.items()):
            names = []
            for n in nodes[:15]:  # Cap per type
                name = graph.nodes[n].get("name", n)
                names.append(name)
            text_parts.append(f"  {label}: {', '.join(names)}")

        # Add key relationships
        rel_types = defaultdict(int)
        for edge in internal_edges:
            rel_types[edge["type"]] += 1
        if rel_types:
            rel_summary = ", ".join(f"{t}: {c}" for t, c in rel_types.items())
            text_parts.append(f"  Relationships: {rel_summary}")

        text = "\n".join(text_parts)

        report = {
            "id": f"community_{comm_id}",
            "community_id": comm_id,
            "members": members,
            "members_by_type": dict(members_by_type),
            "internal_edges": internal_edges[:50],  # Cap edges
            "text": text,
            "summary": "",  # Will be filled by LLM summarization
            "size": len(members),
        }
        reports.append(report)

    return reports


def summarize_communities(reports: List[Dict], llm_fn=None) -> List[Dict]:
    """
    Generate natural language summaries for each community using the LLM.

    Args:
        reports: Community reports from build_community_reports()
        llm_fn: Function that takes a prompt string and returns response text.
                 If None, uses a rule-based summary instead.

    Returns:
        Updated reports with 'summary' field populated
    """
    for report in reports:
        members_by_type = report["members_by_type"]
        text = report["text"]

        if llm_fn is not None:
            # Use LLM to generate a rich summary
            prompt = f"""You are an expert at summarizing academic department information.
Below is a cluster of related entities from the IIT Jammu Electrical Engineering Department's knowledge graph.

{text}

Write a concise 2-4 sentence summary describing what this group represents.
Focus on: who the key people are, what research areas they work on, and how they are connected.
Be specific and factual — do not invent information.

Summary:"""
            try:
                summary = llm_fn(prompt)
                report["summary"] = summary.strip()
                report["text"] = f"{report['text']}\n\nSummary: {summary.strip()}"
                logger.info(f"Summarized community {report['community_id']} ({report['size']} members)")
            except Exception as e:
                logger.warning(f"LLM summarization failed for community {report['community_id']}: {e}")
                report["summary"] = _rule_based_summary(members_by_type)
        else:
            # Rule-based fallback
            report["summary"] = _rule_based_summary(members_by_type)
            report["text"] = f"{report['text']}\n\nSummary: {report['summary']}"

    return reports


def _rule_based_summary(members_by_type: Dict[str, List]) -> str:
    """Generate a simple rule-based summary when LLM is unavailable."""
    parts = []

    faculty = members_by_type.get("Faculty", [])
    students = members_by_type.get("PhDStudent", [])
    areas = members_by_type.get("ResearchArea", [])

    if faculty:
        parts.append(f"This group includes {len(faculty)} faculty member(s): {', '.join(faculty[:5])}")
    if students:
        parts.append(f"{len(students)} PhD student(s)")
    if areas:
        parts.append(f"working in areas like {', '.join(areas[:3])}")

    return ". ".join(parts) + "." if parts else "A cluster of related entities."


def save_communities(reports: List[Dict], partition: Dict[str, int],
                     output_dir: str = DATA_DIR):
    """Save community data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    comm_path = os.path.join(output_dir, "communities.json")
    with open(comm_path, "w", encoding="utf-8") as f:
        json.dump({
            "partition": partition,
            "reports": reports
        }, f, indent=2)

    logger.info(f"Communities saved to {comm_path}")


def load_communities(data_dir: str = DATA_DIR):
    """Load community data from disk."""
    comm_path = os.path.join(data_dir, "communities.json")
    with open(comm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["partition"], data["reports"]
