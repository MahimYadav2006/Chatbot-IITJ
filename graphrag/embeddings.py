"""
Embedding Engine for GraphRAG.
Generates embeddings and FAISS index for chunks, entities, communities.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EMBEDDING_MODEL = "all-mpnet-base-v2"


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Default to CPU to avoid CUDA conflicts/OOMs with Ollama which uses the GPU.
        # User can set EMBEDDING_DEVICE="cuda" to force GPU usage.
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        
        logger.info(f"Using device '{device}' for SentenceTransformer")
        self.model = SentenceTransformer(self.model_name, device=device)
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dim: {dim}")

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        self._load_model()
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size,
                show_progress_bar=show_progress, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            if "out of memory" in str(e).lower() and getattr(self.model, "device", None) and self.model.device.type == "cuda":
                logger.warning("CUDA out of memory during encoding. Falling back to CPU.")
                import torch
                self.model.to("cpu")
                self.model.device = torch.device("cpu")
                embeddings = self.model.encode(texts, batch_size=batch_size,
                    show_progress_bar=show_progress, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            raise e

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1, show_progress=False)[0]

    def build_index(self, chunks: List[Dict], entity_descriptions: List[Dict],
                    community_summaries: List[Dict] = None, dept_code: str = "ee"):
        import faiss
        all_items, all_texts = [], []

        for chunk in chunks:
            all_items.append({"id": chunk["id"], "type": "chunk", "department": dept_code,
                "text": chunk["text"][:1000], "metadata": chunk.get("metadata", {})})
            all_texts.append(chunk["text"][:1000])

        for entity in entity_descriptions:
            all_items.append({"id": entity["id"], "type": "entity", "department": dept_code,
                "text": entity["text"], "metadata": entity.get("metadata", {})})
            all_texts.append(entity["text"])

        if community_summaries:
            for comm in community_summaries:
                all_items.append({"id": comm["id"], "type": "community", "department": dept_code,
                    "text": comm["text"], "metadata": comm.get("metadata", {})})
                all_texts.append(comm["text"])

        logger.info(f"Encoding {len(all_texts)} items...")
        embeddings = self.encode(all_texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = all_items
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 10,
               type_filter: Optional[str] = None, department_filter: Optional[str] = None,
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        query_vec = self.encode_single(query).reshape(1, -1)
        # When filtering by type or department, search ALL vectors
        search_k = self.index.ntotal if (type_filter or department_filter) else top_k
        scores, indices = self.index.search(query_vec, search_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < min_score:
                continue
            item = self.metadata[idx]
            if type_filter and item["type"] != type_filter:
                continue
            if department_filter and item.get("department") != department_filter:
                continue
            results.append((item, float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self, output_dir: str = DATA_DIR):
        import faiss
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, "embeddings.faiss"))
        with open(os.path.join(output_dir, "embeddings_meta.json"), "w") as f:
            json.dump(self.metadata, f)
        logger.info(f"Embeddings saved to {output_dir}")

    def load(self, data_dir: str = DATA_DIR):
        import faiss
        self.index = faiss.read_index(os.path.join(data_dir, "embeddings.faiss"))
        with open(os.path.join(data_dir, "embeddings_meta.json"), "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")


def create_entity_descriptions(graph) -> List[Dict]:
    """Create rich text descriptions for each entity for embedding search."""
    descriptions = []

    # Get department configuration if available
    dept_code = "ee"
    for _, data in graph.nodes(data=True):
        if data.get("department"):
            dept_code = data["department"]
            break

    from departments import get_department
    try:
        dept_config = get_department(dept_code)
        dept_name = dept_config["name"]
        dept_full_name = dept_config["full_name"]
    except Exception:
        dept_name = "Electrical Engineering"
        dept_full_name = "Department of Electrical Engineering"

    for node_id, data in graph.nodes(data=True):
        label = data.get("label", "")
        if label in ("TextChunk", "Document"):
            continue

        parts = []

        if label == "Faculty":
            name = data.get("name", node_id)
            if data.get("is_hod"):
                parts.append(f"{name} is the Head of Department (HoD) of the {dept_full_name} at IIT Jammu.")
                parts.append("As HoD, they lead the department's academic and research activities.")
            else:
                parts.append(f"{name} is a faculty member in the {dept_full_name} at IIT Jammu.")
            if data.get("designation"):
                parts.append(f"Designation: {data['designation']}.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            if data.get("education"):
                parts.append(f"Education: {data['education'][:200]}.")

            out_edges = list(graph.out_edges(node_id, data=True))
            areas = [graph.nodes[t].get('name', t) for _, t, d in out_edges if d.get('type') == 'RESEARCHES_IN']
            if areas:
                parts.append(f"Research areas: {', '.join(areas[:8])}.")
            
            in_edges = list(graph.in_edges(node_id, data=True))
            students = [s for s, _, d in in_edges if d.get('type') == 'SUPERVISED_BY']
            if students:
                parts.append(f"Supervises PhD students: {', '.join(students[:8])}.")

            patents = [graph.nodes[t].get('title', t) for _, t, d in out_edges if d.get('type') == 'INVENTED']
            if patents:
                parts.append(f"Patents: {', '.join(patents[:3])}.")

            startups = [graph.nodes[t].get('name', t) for _, t, d in out_edges if d.get('type') == 'MENTORED_STARTUP']
            if startups:
                parts.append(f"Mentored startups: {', '.join(startups)}.")

        elif label == "PhDStudent":
            name = data.get("name", node_id)
            parts.append(f"{name} is a PhD student in the {dept_name} Department at IIT Jammu.")
            if data.get("research_area"):
                parts.append(f"Research topic: {data['research_area']}.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            
            out_edges = list(graph.out_edges(node_id, data=True))
            supervisors = [graph.nodes[t].get('name', t) for _, t, d in out_edges if d.get('type') == 'SUPERVISED_BY']
            if supervisors:
                parts.append(f"Supervised by: {', '.join(supervisors)}.")

        elif label == "ResearchArea":
            name = data.get("name", node_id)
            category = data.get("category", "")
            parts.append(f"{name} is a research area in the {dept_name} Department at IIT Jammu.")
            if category:
                parts.append(f"Falls under the {category} category.")

        elif label == "ResearchCategory":
            name = data.get("name", node_id)
            parts.append(f"{name} is a major research category in the {dept_name} Department at IIT Jammu.")
            in_edges = list(graph.in_edges(node_id, data=True))
            sub_areas = [graph.nodes[s].get('name', s) for s, _, d in in_edges if d.get('type') == 'BELONGS_TO_CATEGORY']
            if sub_areas:
                parts.append(f"Includes: {', '.join(sub_areas[:8])}.")

        elif label == "Project":
            parts.append(f"Funded project: {data.get('title', node_id)}.")
            if data.get("agency"):
                parts.append(f"Funded by: {data['agency']}.")

        elif label == "Patent":
            parts.append(f"Patent: {data.get('title', node_id)}.")
            if data.get("application_no"):
                parts.append(f"Application number: {data['application_no']}.")

        elif label == "Startup":
            parts.append(f"Startup: {data.get('name', node_id)}.")
            if data.get("description"):
                parts.append(data["description"])

        elif label == "Lab":
            parts.append(f"{data.get('name', node_id)} is a laboratory in the {dept_name} Department at IIT Jammu.")

        elif label == "Department":
            parts.append(f"{dept_full_name} at IIT Jammu.")
            parts.append(f"The department offers BTech, MTech, and PhD programmes in {dept_name}.")
            parts.append("Head of Department, HoD, faculty, research areas, labs, students.")

        elif label == "Programme":
            parts.append(f"{data.get('name', node_id)} is an academic programme at IIT Jammu {dept_name} Department.")

        elif label == "FundingAgency":
            parts.append(f"{data.get('name', node_id)} is a funding agency.")

        elif label == "Placement":
            parts.append(f"{data.get('name', node_id)} — placement record ({data.get('placement_type', 'N/A')}).")

        elif label == "ExternalPerson":
            name = data.get("name", node_id)
            parts.append(f"{name} is an external collaborator/supervisor associated with the {dept_name} Department at IIT Jammu.")
            in_edges = list(graph.in_edges(node_id, data=True))
            students = [s for s, _, d in in_edges if d.get('type') == 'SUPERVISED_BY']
            if students:
                parts.append(f"Co-supervises PhD students: {', '.join(students[:5])}.")

        else:
            parts.append(f"{data.get('name', node_id)} ({label})")

        if parts:
            descriptions.append({
                "id": node_id,
                "text": " ".join(parts),
                "metadata": {"label": label, "name": data.get("name", node_id)}
            })

    return descriptions
