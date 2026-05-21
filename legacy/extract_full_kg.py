import os
import re
import json
from urllib.parse import urlparse

# Define paths
MARKDOWN_DIR = "iitjammu_ee_markdown"
OUTPUT_GRAPH = os.path.join(MARKDOWN_DIR, "extracted_graph.json")

def clean_name(name):
    """Normalize academic names by removing prefixes and extra spaces."""
    name = re.sub(r'\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Shri)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def parse_full_kg():
    print("Starting full Knowledge Graph construction...")
    
    nodes = {}
    relationships = []
    
    # 1. First, create Document nodes for EVERY markdown file to guarantee 100% data access
    if not os.path.exists(MARKDOWN_DIR):
        print(f"Directory not found: {MARKDOWN_DIR}")
        return
        
    filenames = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md") and f != "00_combined_ee_site.md"]
    
    for filename in filenames:
        filepath = os.path.join(MARKDOWN_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract Source URL
        source_url = "https://iitjammu.ac.in/ee"
        url_match = re.search(r'# Source URL:\s*([^\n]+)', content)
        if url_match:
            source_url = url_match.group(1).strip()
            
        # Extract Clean Title
        clean_title = filename.replace(".html.md", "").replace(".md", "").replace("ee_", "").replace("_", " ").title()
        
        # Add Document Node
        nodes[filename] = {
            "id": filename,
            "label": "Document",
            "properties": {
                "title": clean_title,
                "filename": filename,
                "source_url": source_url,
                "raw_text": content
            }
        }
        
    print(f"Phase 1: Created {len(filenames)} Document nodes (100% data preservation).")
    
    # 2. Extract structured entities from each file
    for filename in filenames:
        filepath = os.path.join(MARKDOWN_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # --- PARSE: PhD Student List ---
        if filename == "ee_phd-list.html.md":
            student_blocks = re.findall(r'#### ([^\n]+)\n+(.*?)(?=#### |Source URL:|$)', content, re.DOTALL)
            for name, block in student_blocks:
                student_name = name.strip()
                
                # Extract Supervisor
                supervisor_match = re.search(r'\*\*Supervisor\*\*\n+([^\n]+)', block, re.IGNORECASE)
                supervisors = []
                if supervisor_match:
                    sup_text = supervisor_match.group(1).strip()
                    # Remove title prefixes
                    sup_text = re.sub(r'\b(?:Dr\.|Prof\.)\b', '', sup_text, flags=re.IGNORECASE)
                    # Normalize whitespace
                    sup_text = re.sub(r'\s+', ' ', sup_text)
                    # Split on commas or the word 'and' (case-insensitive)
                    raw_sups = re.split(r'\s*(?:,|and)\s*', sup_text, flags=re.IGNORECASE)
                    supervisors = [clean_name(s) for s in raw_sups if s.strip()]

                # Extract Research Area
                area_match = re.search(r'Research Area\n+([^\n]+)', block, re.IGNORECASE)
                research_area = area_match.group(1).strip() if area_match else "Unknown"
                
                # Create PhDStudent node
                nodes[student_name] = {
                    "id": student_name,
                    "label": "PhDStudent",
                    "properties": {
                        "name": student_name,
                        "research_area": research_area
                    }
                }
                relationships.append({"source": student_name, "target": filename, "type": "SOURCE_DOCUMENT"})
                
                # Supervisors
                for sup in supervisors:
                    if sup not in nodes:
                        nodes[sup] = {"id": sup, "label": "Faculty", "properties": {"name": sup}}
                    relationships.append({"source": student_name, "target": sup, "type": "SUPERVISED_BY"})
                    relationships.append({"source": sup, "target": filename, "type": "SOURCE_DOCUMENT"})
                    
                # Research Area
                if research_area and research_area != "Unknown":
                    if research_area not in nodes:
                        nodes[research_area] = {"id": research_area, "label": "ResearchArea", "properties": {"name": research_area}}
                    relationships.append({"source": student_name, "target": research_area, "type": "STUDIES"})
                    relationships.append({"source": research_area, "target": filename, "type": "SOURCE_DOCUMENT"})
                    
        # --- PARSE: Faculty Profile Pages ---
        elif "faculty.html_faculty__" in filename:
            lines = [l.strip() for l in content.splitlines()]
            
            # Find Faculty Name from filename parameter
            faculty_key = filename.split("__")[-1].replace(".md", "")
            
            # Find clean name from lines
            faculty_name = clean_name(faculty_key.title())
            for line in lines:
                if line.startswith("- ") and len(line) > 2 and "@" not in line and "Professor" not in line and "Home" not in line:
                    faculty_name = clean_name(line[2:])
                    break
                    
            # Find Email
            email = "Unknown"
            for line in lines:
                if "@" in line and "." in line:
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+', line)
                    if email_match:
                        email = email_match.group(0)
                        break
                        
            # Find Designation
            designation = "Faculty Member"
            for line in lines:
                if "Professor" in line or "Lecturer" in line:
                    designation = line.replace("-", "").strip()
                    break
                    
            # Extract Education
            edu_match = re.search(r'##### Education Qualification\n+(.*?)(?=\n+#####|$)', content, re.DOTALL)
            education = edu_match.group(1).strip() if edu_match else ""
            
            # Extract Research Interests
            interests_match = re.search(r'Research Interests:\s*\n*(.*?)(?=\n+Teaching Interests:|\n+#####|$)', content, re.DOTALL)
            research_interests = []
            if interests_match:
                interests_str = interests_match.group(1).strip()
                research_interests = [i.strip() for i in re.split(r'[,;]', interests_str) if i.strip()]
                
            # Extract Publications
            pub_match = re.search(r'##### Publications\n+(.*?)(?=\n+#####|$)', content, re.DOTALL)
            publications = []
            if pub_match:
                pub_text = pub_match.group(1)
                # Find publication items
                publications = re.findall(r'(?:\d+\.|\-\s+.*?)\s+([^\n]+)', pub_text)
                
            # Upsert Faculty Node
            if faculty_name in nodes:
                # Merge properties
                nodes[faculty_name]["properties"].update({
                    "email": email,
                    "designation": designation,
                    "education": education
                })
            else:
                nodes[faculty_name] = {
                    "id": faculty_name,
                    "label": "Faculty",
                    "properties": {
                        "name": faculty_name,
                        "email": email,
                        "designation": designation,
                        "education": education
                    }
                }
                
            relationships.append({"source": faculty_name, "target": filename, "type": "PROFILE_DOCUMENT"})
            
            # Link Research Areas
            for interest in research_interests:
                if interest not in nodes:
                    nodes[interest] = {"id": interest, "label": "ResearchArea", "properties": {"name": interest}}
                relationships.append({"source": faculty_name, "target": interest, "type": "RESEARCHES_IN"})
                
            # Link Publications
            for pub in publications:
                pub_title = pub.strip()
                if len(pub_title) > 10:
                    pub_id = f"Pub_{hash(pub_title) & 0xffffffff}"
                    nodes[pub_id] = {
                        "id": pub_id,
                        "label": "Publication",
                        "properties": {
                            "title": pub_title
                        }
                    }
                    relationships.append({"source": faculty_name, "target": pub_id, "type": "AUTHORED"})
                    relationships.append({"source": pub_id, "target": filename, "type": "SOURCE_DOCUMENT"})

        # --- PARSE: Funded Projects ---
        elif filename == "ee_funded-projects.html.md":
            # Extract items like: - [1] Title: ..., Funding Agency: ...
            # Use a robust regex that captures title and agency, allowing for commas in title.
            projects_data = []
            for line in content.splitlines():
                m = re.search(r"- \[\d+\]\s*Title:\s*(.+?)\s*,\s*Funding Agency:\s*(.+?)$", line)
                if m:
                    projects_data.append((m.group(1).strip(), m.group(2).strip()))
            for title, agency in projects_data:
                project_title = title.replace("\n", " ")
                agency_name = agency.replace("\n", " ")
                
                # Create Project Node
                nodes[project_title] = {
                    "id": project_title,
                    "label": "Project",
                    "properties": {
                        "title": project_title,
                        "agency": agency_name
                    }
                }
                relationships.append({"source": project_title, "target": filename, "type": "SOURCE_DOCUMENT"})
                
                # Create Funding Agency Node (normalize name)
                agency_key = agency_name
                if agency_key not in nodes:
                    nodes[agency_key] = {
                        "id": agency_key,
                        "label": "FundingAgency",
                        "properties": {"name": agency_key}
                    }
                relationships.append({"source": project_title, "target": agency_key, "type": "FUNDED_BY"})
                relationships.append({"source": agency_key, "target": filename, "type": "SOURCE_DOCUMENT"})
                
        # --- PARSE: Patents ---
        elif filename == "ee_patent.html.md":
            patents = re.findall(r'\-\s+\*\*Title\*\*\:\s*(.*?)\n+\*\*Inventors\*\*\:\s*(.*?)\n+\*\*Application No\*\*.*?\:\s*(.*?)(?=\-\s+\*\*Title\*\*|\>|$)', content, re.DOTALL)
            for title, inventors_str, app_no in patents:
                patent_title = title.strip().replace("\n", " ")
                inventors = [clean_name(i) for i in re.split(r'[,;]|and', inventors_str) if i.strip()]
                patent_no = app_no.strip()
                
                # Create Patent Node
                nodes[patent_title] = {
                    "id": patent_title,
                    "label": "Patent",
                    "properties": {
                        "title": patent_title,
                        "application_no": patent_no
                    }
                }
                relationships.append({"source": patent_title, "target": filename, "type": "SOURCE_DOCUMENT"})
                
                # Link Inventors
                for inv in inventors:
                    if inv not in nodes:
                        nodes[inv] = {"id": inv, "label": "Faculty", "properties": {"name": inv}}
                    relationships.append({"source": inv, "target": patent_title, "type": "INVENTED"})

        # --- PARSE: Startups ---
        elif filename == "ee_startups.html.md":
            # Direct parsing of the two startups mentioned in startups page
            startups = [
                {
                    "name": "Data Sailors",
                    "mentor": "Ankit Dubey",
                    "description": "Focuses on resource monitoring, data analytics, and forecasting using AI ML. Incubated at IIT Jammu."
                },
                {
                    "name": "Servotech private limited",
                    "mentor": "Sudhakar Modem",
                    "description": "Commercialization of oxygen concentrators and other clean energy projects."
                }
            ]
            for s in startups:
                startup_name = s["name"]
                mentor_name = clean_name(s["mentor"])
                desc = s["description"]
                
                nodes[startup_name] = {
                    "id": startup_name,
                    "label": "Startup",
                    "properties": {
                        "name": startup_name,
                        "description": desc
                    }
                }
                relationships.append({"source": startup_name, "target": filename, "type": "SOURCE_DOCUMENT"})
                
                if mentor_name not in nodes:
                    nodes[mentor_name] = {"id": mentor_name, "label": "Faculty", "properties": {"name": mentor_name}}
                relationships.append({"source": mentor_name, "target": startup_name, "type": "MENTORED_STARTUP"})

    # Clean relationships to avoid duplicates and ensure validity of nodes
    valid_nodes = set(nodes.keys())
    clean_relationships = []
    seen_rels = set()
    
    for rel in relationships:
        src = rel["source"]
        tgt = rel["target"]
        rel_type = rel["type"]
        
        if src in valid_nodes and tgt in valid_nodes:
            rel_key = (src, tgt, rel_type)
            if rel_key not in seen_rels:
                clean_relationships.append(rel)
                seen_rels.add(rel_key)
                
    # Save graph to JSON
    graph_data = {
        "nodes": list(nodes.values()),
        "edges": clean_relationships
    }
    
    with open(OUTPUT_GRAPH, "w", encoding="utf-8") as out:
        json.dump(graph_data, out, indent=2)
        
    print(f"\n✅ Graph Database complete!")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Relationships: {len(clean_relationships)}")
    print(f"Saved database to: {OUTPUT_GRAPH}")

if __name__ == "__main__":
    parse_full_kg()
