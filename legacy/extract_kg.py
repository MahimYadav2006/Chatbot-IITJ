import os
import re
import json

def parse_phd_students(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return [], []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find student sections
    # Pattern: #### Name followed by Supervisor and Research Area
    student_blocks = re.findall(r'#### ([^\n]+)\n+(.*?)(?=#### |Source URL:|$)', content, re.DOTALL)
    
    nodes = {}
    relationships = []
    
    for name, block in student_blocks:
        student_name = name.strip()
        
        # Extract Supervisor
        supervisor_match = re.search(r'\*\*Supervisor\*\*\n+([^\n]+)', block, re.IGNORECASE)
        supervisors = []
        if supervisor_match:
            sup_text = supervisor_match.group(1).strip()
            # Clean up academic titles
            sup_text = re.sub(r'\b(?:Dr\.|Prof\.)\b', '', sup_text, flags=re.IGNORECASE)
            # Split by "And", "and", "And ", or ","
            raw_sups = re.split(r'\b(?:and|,)\b', sup_text, flags=re.IGNORECASE)
            supervisors = [s.strip() for s in raw_sups if s.strip()]

        # Extract Research Area
        area_match = re.search(r'Research Area\n+([^\n]+)', block, re.IGNORECASE)
        research_area = area_match.group(1).strip() if area_match else "Unknown"

        # 1. Add Student Node
        nodes[student_name] = {
            "id": student_name,
            "label": "PhDStudent",
            "properties": {
                "name": student_name,
                "research_area": research_area
            }
        }

        # 2. Add Faculty Nodes & Supervised Edges
        for sup in supervisors:
            # Clean double spaces
            sup = re.sub(r'\s+', ' ', sup)
            if sup not in nodes:
                nodes[sup] = {
                    "id": sup,
                    "label": "Faculty",
                    "properties": {"name": sup}
                }
            relationships.append({
                "source": student_name,
                "target": sup,
                "type": "SUPERVISED_BY"
            })

        # 3. Add Research Area Node & Studies Edge
        if research_area and research_area != "Unknown":
            if research_area not in nodes:
                nodes[research_area] = {
                    "id": research_area,
                    "label": "ResearchArea",
                    "properties": {"name": research_area}
                }
            relationships.append({
                "source": student_name,
                "target": research_area,
                "type": "STUDIES"
            })

    return list(nodes.values()), relationships

if __name__ == "__main__":
    filepath = "iitjammu_ee_markdown/ee_phd-list.html.md"
    nodes, edges = parse_phd_students(filepath)
    
    graph_data = {
        "nodes": nodes,
        "edges": edges
    }
    
    print(f"Extracted {len(nodes)} Nodes and {len(edges)} Relationships!")
    
    # Save graph as JSON
    with open("iitjammu_ee_markdown/extracted_graph.json", "w", encoding="utf-8") as out:
        json.dump(graph_data, out, indent=2)
    print("Graph successfully saved to 'iitjammu_ee_markdown/extracted_graph.json'!")
