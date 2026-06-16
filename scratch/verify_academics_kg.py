import os
import pickle
import networkx as nx
from collections import Counter

def main():
    data_dir = "/home/c3i/chatbot/data/sections/academics"
    pkl_path = os.path.join(data_dir, "graph.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: graph.pkl not found at {pkl_path}")
        return

    print("Loading Academics Section Graph...")
    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)
    print(f"Graph loaded successfully: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Node label distribution
    labels = [data.get("label", "Unknown") for node, data in graph.nodes(data=True)]
    print("\nNode Label Distribution:")
    for label, count in Counter(labels).items():
        print(f"  - {label}: {count}")

    # Specific nodes listing
    programs = [data.get("name") for node, data in graph.nodes(data=True) if data.get("label") == "AcademicProgram"]
    specializations = [data.get("name") for node, data in graph.nodes(data=True) if data.get("label") == "Specialization"]
    courses = [data.get("name") for node, data in graph.nodes(data=True) if data.get("label") == "Course"]
    buckets = [data.get("name") for node, data in graph.nodes(data=True) if data.get("label") == "ElectiveBucket"]

    print(f"\nTotal Academic Programs: {len(programs)}")
    for prog in sorted(programs)[:15]:
        print(f"  - Program: {prog}")
    if len(programs) > 15:
        print("    ...")

    print(f"\nTotal Specializations/Minors: {len(specializations)}")
    for spec in sorted(specializations)[:15]:
        print(f"  - Specialization: {spec}")
    if len(specializations) > 15:
        print("    ...")

    print(f"\nTotal Courses: {len(courses)}")
    for course in sorted(courses)[:15]:
        print(f"  - Course: {course}")
    if len(courses) > 15:
        print("    ...")

    print(f"\nTotal Elective Buckets: {len(buckets)}")
    for bucket in sorted(buckets)[:15]:
        print(f"  - Bucket: {bucket}")
    if len(buckets) > 15:
        print("    ...")

    # Edge label distribution
    edge_types = [data.get("type", "Unknown") for u, v, data in graph.edges(data=True)]
    print("\nEdge Type Distribution:")
    for type_val, count in Counter(edge_types).items():
        print(f"  - {type_val}: {count}")

    # Check for some specific relations
    sample_edges = [(u, v, data) for u, v, data in graph.edges(data=True) if data.get("type") == "OFFERS_COURSE"]
    print(f"\nSample OFFERS_COURSE edges: {len(sample_edges)}")
    for u, v, data in sample_edges[:15]:
        u_name = graph.nodes[u].get("name", u)
        v_name = graph.nodes[v].get("name", v)
        print(f"  - {u_name} -> OFFERS_COURSE -> {v_name} (sem={data.get('semester')}, cat={data.get('category')}, bucket={data.get('bucket')})")

if __name__ == "__main__":
    main()
