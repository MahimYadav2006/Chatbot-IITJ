"""
Section Knowledge Graph Builder for IIT Jammu Sections.
Parses section markdown files and constructs a NetworkX DiGraph with entities,
relationships, and clean text chunks.
"""

import os
import re
import json
import pickle
import logging
from collections import defaultdict
import networkx as nx

from departments import SECTIONS, get_section_markdown_dir, get_section_data_dir
from graphrag.kg_builder import (
    clean_content_for_chunks,
    smart_chunk_text,
    EntityResolver,
    normalize_name,
    clean_admin_member_name,
    _deobfuscate_email_text,
    _strip_markdown_emphasis,
    _strip_markdown_link
)

logger = logging.getLogger(__name__)

class SectionKGBuilder:
    def __init__(self, section_code: str, markdown_dir: str = None):
        self.section_code = section_code
        self.section_config = SECTIONS[section_code]
        self.markdown_dir = markdown_dir or get_section_markdown_dir(section_code)
        self.graph = nx.DiGraph()
        self.resolver = EntityResolver()
        self.chunks = []

    def _add_node(self, node_id: str, label: str, **properties):
        # Prefix node ID with section_code to keep it unique, unless it starts with doc: or chunk_
        if not node_id.startswith(f"{self.section_code}:") and not node_id.startswith("doc:") and not node_id.startswith("chunk_"):
            node_id = f"{self.section_code}:{node_id}"
        
        properties["section"] = self.section_code
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(properties)
        else:
            self.graph.add_node(node_id, label=label, **properties)
        return node_id

    def _add_edge(self, source: str, target: str, rel_type: str, **properties):
        if not source.startswith(f"{self.section_code}:") and not source.startswith("doc:") and not source.startswith("chunk_"):
            source = f"{self.section_code}:{source}"
        if not target.startswith(f"{self.section_code}:") and not target.startswith("doc:") and not target.startswith("chunk_"):
            target = f"{self.section_code}:{target}"
            
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return
        self.graph.add_edge(source, target, type=rel_type, **properties)

    def _create_document_node(self, filename: str, content: str) -> str:
        source_url = self.section_config.get("base_url", "https://iitjammu.ac.in")
        url_match = re.search(r'# Source URL:\s*([^\n]+)', content)
        if url_match:
            source_url = url_match.group(1).strip()

        clean_title = (filename.replace(".html.md", "").replace(".md", "")
            .replace(f"{self.section_code}_", "").replace("_", " ").title())

        doc_id = self._add_node(f"doc:{filename}", "Document", title=clean_title,
            filename=filename, source_url=source_url)

        # Clean content before chunking — remove boilerplate
        clean_content = clean_content_for_chunks(content)
        chunk_items = smart_chunk_text(clean_content)
        if chunk_items:
            self.graph.nodes[doc_id]["chunk_strategy"] = chunk_items[0]["meta"].get("strategy", "unknown")
        
        for idx, chunk_item in enumerate(chunk_items):
            chunk_text_str = chunk_item["text"]
            chunk_meta = chunk_item.get("meta", {})
            if len(chunk_text_str.strip()) < 30:
                continue
            chunk_id = f"chunk_{filename}_{idx}"
            self._add_node(chunk_id, "TextChunk", text=chunk_text_str,
                doc_filename=filename, chunk_index=idx, source_url=source_url,
                chunk_strategy=chunk_meta.get("strategy", "unknown"))
            self._add_edge(doc_id, chunk_id, "HAS_CHUNK")
            self.chunks.append((chunk_id, chunk_text_str, {
                "doc": filename, "url": source_url,
                "title": clean_title, "chunk_idx": idx,
                "chunk_strategy": chunk_meta.get("strategy", "unknown"),
                "chunk_meta": chunk_meta,
            }))
        return doc_id

    def _parse_people_list(self, filename: str, content: str, doc_id: str):
        """Parse the academics_people-list.md and accounts/e2 people lists.
        Format:
        #### Ajay Singh
        ##### **Designation**
        ###### Associate Dean: Curriculum (PG)
        ##### adpg.acad@iitjammu.ac.in
        """
        lines = [l.strip() for l in content.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#### "):
                name = line[5:].strip()
                designation = ""
                email = ""
                
                # Scan ahead for designation and email
                j = i + 1
                while j < len(lines) and not lines[j].startswith("#### "):
                    sub_line = lines[j]
                    if sub_line.startswith("###### "):
                        designation = sub_line[7:].strip()
                    elif "@" in sub_line or "at" in sub_line.lower():
                        email = _deobfuscate_email_text(sub_line.replace("#####", "").strip())
                    j += 1
                
                # Resolve name
                resolved_name = self.resolver.resolve(name)
                self._add_node(resolved_name, "SectionPerson",
                               name=resolved_name,
                               designation=designation,
                               email=email,
                               source_file=filename)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                i = j - 1
            i += 1

    def _parse_counselling_team(self, filename: str, content: str, doc_id: str):
        """Parse counselling team tables from counselling_team.md."""
        # Find lines with tables
        lines = [l.strip() for l in content.splitlines()]
        for line in lines:
            if "|" in line and "Coordinator" in line:
                # E.g. | Dr Sanat Kumar Tiwari | Coordinator - Counselling Services | sanat.tiwari@iitjammu.ac.in Phone: 0191-257-0281, Office: 01AC-6-32 |
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 3:
                    name = clean_admin_member_name(cells[0])
                    designation = cells[1]
                    contact = cells[2]
                    email_match = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', contact)
                    email = email_match.group(0) if email_match else ""
                    phone_match = re.search(r'Phone:\s*([\d\-]+)', contact)
                    phone = phone_match.group(1) if phone_match else ""
                    
                    resolved_name = self.resolver.resolve(name)
                    self._add_node(resolved_name, "SectionHead",
                                   name=resolved_name,
                                   designation=designation,
                                   email=email,
                                   phone=phone,
                                   source_file=filename)
                    self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
            elif "|" in line and ("himanshi.singh" in line or "nandita.sharma" in line):
                # E.g. | **Himanshi Singh** Language Proficiency... | himanshi.singh@iitjammu.ac.in | 9797894944 | Level 108... |
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 3:
                    name_raw = cells[0].split("Language")[0].replace("**", "").strip()
                    name = clean_admin_member_name(name_raw)
                    email = cells[1]
                    phone = cells[2]
                    office = cells[3] if len(cells) > 3 else ""
                    
                    resolved_name = self.resolver.resolve(name)
                    self._add_node(resolved_name, "Counselor",
                                   name=resolved_name,
                                   designation="Institute Counselor",
                                   email=email,
                                   phone=phone,
                                   office=office,
                                   source_file=filename)
                    self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")

    def _parse_counselor_profiles(self, filename: str, content: str, doc_id: str):
        """Parse bio information from counselling_know-your-counselors.md."""
        # Split by counselor name blocks
        # E.g. **Himanshi Singh** followed by paragraphs
        blocks = re.split(r'\*\*([^*]+)\*\*', content)
        # The split returns: [text before first match, name1, text1, name2, text2, ...]
        for i in range(1, len(blocks), 2):
            name = clean_admin_member_name(blocks[i].strip())
            bio = blocks[i+1].strip() if i+1 < len(blocks) else ""
            if name and bio:
                resolved_name = self.resolver.resolve(name)
                self._add_node(resolved_name, "Counselor",
                               name=resolved_name,
                               bio=bio,
                               source_file=filename)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")

    def _parse_di_team(self, filename: str, content: str, doc_id: str):
        """Parse team hierarchy from di_team.html.md."""
        lines = [l.strip() for l in content.splitlines()]
        
        # Deans & ADs
        # #### Dr. Badri Narayan Subudhi
        # Dean, DI](url)
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#### "):
                name = line[5:].strip()
                # Resolve designation from next lines
                designation = ""
                j = i + 1
                while j < len(lines) and not lines[j].startswith("#### ") and not lines[j].startswith("## "):
                    sub_line = lines[j]
                    if sub_line:
                        # Clean up links like Dean, DI](https...)
                        designation = re.sub(r'\]\(https?:.*$', '', sub_line).strip()
                        break
                    j += 1
                
                resolved_name = self.resolver.resolve(name)
                # Assign role labels
                if "Dean" in designation:
                    label = "SectionHead" if "Associate" not in designation else "SectionPerson"
                else:
                    label = "SectionPerson"
                    
                self._add_node(resolved_name, label,
                               name=resolved_name,
                               designation=designation,
                               source_file=filename)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                i = j - 1
            i += 1

    def _parse_section_contact(self, filename: str, content: str, doc_id: str):
        """Parse contact details from contact markdown files."""
        # Find emails, phones, address, and hours
        email = ""
        phone = ""
        hours = ""
        address = ""
        
        emails = re.findall(r'[\w\.\-]+@[\w\.\-]+\.\w+', content)
        if emails:
            email = ", ".join(sorted(list(set(emails))))
            
        phones = re.findall(r'(?:Phone|Telephone|Tel|Mobile)\s*[:\s]*([+\d\s\-()]{8,})', content, re.IGNORECASE)
        if phones:
            phone = ", ".join(sorted(list(set([p.strip() for p in phones]))))
            
        hours_match = re.search(r'(?:Working Hours|Hours|Timing)\s*[:\s]*([^\n]+)', content, re.IGNORECASE)
        if hours_match:
            hours = hours_match.group(1).strip()
            
        address_match = re.search(r'Address\s*[:\s]*([^\n#]+)', content, re.IGNORECASE)
        if address_match:
            address = address_match.group(1).strip()
            
        if self.section_code == "counselling":
            hours = "Monday to Friday: 9:30 AM to 5:30 PM"
            phone = "0191-2570730"
            email = "counselling.services@iitjammu.ac.in"
            
        if self.section_code == "di":
            phone = "0191-2570280"
            email = "c3i@iitjammu.ac.in"
            address = "Digital Infrastructure Office, IIT Jammu, Jagti, Nagrota, J&K"

        if self.section_code == "academics":
            phone = "0191-2570633"
            email = "ar.acad@iitjammu.ac.in"
            
        contact_id = f"contact:{self.section_code}"
        self._add_node(contact_id, "SectionContact",
                       name=f"{self.section_config['name']} Contact Information",
                       email=email,
                       phone=phone,
                       hours=hours,
                       address=address,
                       source_file=filename)
        self._add_edge(contact_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_hod_message(self, filename: str, content: str, doc_id: str):
        """Parse HOD/Dean message file to extract head info."""
        # E.g. academics_hod-message.md, accounts_hod-message.md
        lines = [l.strip() for l in content.splitlines()]
        name = ""
        designation = ""
        
        # Look for #### or ### name / designation
        for i, line in enumerate(lines):
            if line.startswith("### "):
                name = clean_admin_member_name(line[4:])
            elif line.startswith("#### "):
                designation = line[5:].replace("**", "").replace("|", "").strip()
                
        if self.section_code == "academics" and not name:
            name = "Sartaj Ul Hasan"
            designation = "Dean, Academic Programs"
        elif self.section_code == "accounts" and not name:
            name = "Shikha Malhotra"
            designation = "Assistant Registrar, Finance and Accounts"
            
        if name:
            resolved_name = self.resolver.resolve(name)
            self._add_node(resolved_name, "SectionHead",
                           name=resolved_name,
                           designation=designation,
                           is_head=True,
                           source_file=filename)
            self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")

    def build(self) -> nx.DiGraph:
        if not os.path.exists(self.markdown_dir):
            raise FileNotFoundError(f"Markdown directory not found: {self.markdown_dir}")

        filenames = [f for f in os.listdir(self.markdown_dir)
                     if f.endswith(".md") and not f.startswith("00_combined")
                     and not f.endswith(".json")]

        logger.info(f"Processing {len(filenames)} section markdown files...")

        # Parse E2 head Puja Rajyaguru as SectionHead
        if self.section_code == "e2":
            self._add_node("Puja Rajyaguru", "SectionHead",
                           name="Puja Rajyaguru",
                           designation="Assistant Registrar (Establishment II)",
                           is_head=True)

        doc_map = {}
        for filename in sorted(filenames):
            filepath = os.path.join(self.markdown_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id = self._create_document_node(filename, content)
            doc_map[filename] = (doc_id, content)

        for filename, (doc_id, content) in doc_map.items():
            if "people-list" in filename or "team" in filename:
                if self.section_code == "counselling":
                    self._parse_counselling_team(filename, content, doc_id)
                elif self.section_code == "di":
                    self._parse_di_team(filename, content, doc_id)
                else:
                    self._parse_people_list(filename, content, doc_id)
            elif "know-your-counselors" in filename:
                self._parse_counselor_profiles(filename, content, doc_id)
            elif "hod-message" in filename:
                self._parse_hod_message(filename, content, doc_id)
            elif "contact" in filename:
                self._parse_section_contact(filename, content, doc_id)

        # Connect Section Person nodes to Section Head
        section_node = f"section:{self.section_code}"
        self._add_node(section_node, "Section",
                       name=self.section_config["name"],
                       base_url=self.section_config["base_url"])
        
        # Link all extracted persons to the Section
        for node, data in list(self.graph.nodes(data=True)):
            if data.get("label") in ("SectionPerson", "SectionHead", "Counselor"):
                self._add_edge(node, section_node, "BELONGS_TO_SECTION")

        logger.info(f"Section Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def save(self, output_dir: str = None):
        if output_dir is None:
            output_dir = get_section_data_dir(self.section_code)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)
        chunks_data = [{"id": c[0], "text": c[1], "metadata": c[2]} for c in self.chunks]
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        with open(os.path.join(output_dir, "resolver.pkl"), "wb") as f:
            pickle.dump(self.resolver, f)
        logger.info(f"Section Graph saved to {output_dir}")

    @staticmethod
    def load(data_dir: str):
        with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
            graph = pickle.load(f)
        with open(os.path.join(data_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return graph, chunks


def create_section_entity_descriptions(graph, section_code: str) -> list:
    descriptions = []
    from departments import SECTIONS
    section_name = SECTIONS[section_code]["name"]
    for node_id, data in graph.nodes(data=True):
        label = data.get("label", "")
        if label in ("TextChunk", "Document"):
            continue
        parts = []
        name = data.get("name", node_id)
        if isinstance(name, str) and ":" in name and not name.startswith("http"):
            name = name.split(":", 1)[1]
            
        if label == "SectionHead":
            parts.append(f"{name} is the head or coordinator of the {section_name} section at IIT Jammu.")
            if data.get("designation"):
                parts.append(f"Designation: {data['designation']}.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            if data.get("phone"):
                parts.append(f"Phone: {data['phone']}.")
        elif label == "Counselor":
            parts.append(f"{name} is an Institute Counselor in the Counselling section at IIT Jammu.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            if data.get("phone"):
                parts.append(f"Phone: {data['phone']}.")
            if data.get("office"):
                parts.append(f"Office: {data['office']}.")
            if data.get("bio"):
                parts.append(f"Bio/profile: {data['bio']}.")
        elif label == "SectionPerson":
            parts.append(f"{name} is a member of the {section_name} section at IIT Jammu.")
            if data.get("designation"):
                parts.append(f"Designation: {data['designation']}.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
        elif label == "SectionContact":
            parts.append(f"Contact details for {section_name} section at IIT Jammu.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            if data.get("phone"):
                parts.append(f"Phone: {data['phone']}.")
            if data.get("address"):
                parts.append(f"Address: {data['address']}.")
            if data.get("hours"):
                parts.append(f"Working Hours: {data['hours']}.")
        else:
            parts.append(f"{name} ({label}) in {section_name} section.")

        if parts:
            descriptions.append({
                "id": node_id,
                "text": " ".join(parts),
                "metadata": {"label": label, "name": name}
            })
    return descriptions

