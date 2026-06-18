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
from typing import Optional, Dict, List, Any
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

        base_fn = os.path.basename(filename)
        clean_title = (base_fn.replace(".html.md", "").replace(".md", "")
            .replace(f"{self.section_code}_", "").replace("_", " ").title())

        # Extract notification date if present
        date_match = re.search(r'\*\*Date:\*\*\s*([0-9a-zA-Z\s,]+)', content, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None

        doc_id = self._add_node(f"doc:{filename}", "Document", title=clean_title,
            filename=filename, source_url=source_url, notification_date=date_str)

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
            safe_fn = filename.replace("/", "_").replace("\\", "_")
            chunk_id = f"chunk_{safe_fn}_{idx}"
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

    def _parse_committee_document(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        
        # Extract notification date if present
        date_match = re.search(r'\*\*Date:\*\*\s*([0-9a-zA-Z\s,]+)', content, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None
        
        # Determine committee title from first H1 or title
        title = ""
        h1_match = re.search(r'^\s*#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        else:
            title = os.path.basename(filename).replace(".md", "").replace("_", " ").title()
            
        title = re.sub(r'\*+', '', title).strip()
        
        # If it is DPGC or DUGC, we use department sub-headings to partition
        is_dept_committee = "dpgc" in filename.lower() or "dugc" in filename.lower()
        committee_type = "DPGC" if "dpgc" in filename.lower() else ("DUGC" if "dugc" in filename.lower() else "General")
        
        current_dept = None
        in_table = False
        headers = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for department heading
            if is_dept_committee:
                dept_match = re.match(r'^(?:#|##)\s+Department\s+of\s+(.+)$', line_stripped, re.IGNORECASE)
                if dept_match:
                    current_dept = dept_match.group(1).strip()
                    in_table = False
                    continue
            
            if line_stripped.startswith('|'):
                cells = [c.strip() for c in line_stripped.split('|')[1:-1]]
                if not cells:
                    continue
                
                # Check for divider
                if all(re.match(r'^:?-+:?$', c) for c in cells):
                    in_table = True
                    continue
                    
                if not in_table:
                    headers = [c.lower() for c in cells]
                    continue
                
                # Check if this row is metadata (contains Institution, Date, etc.)
                if any(h in headers for h in ("field", "value")) or any(c.lower() in ("institution", "notification number", "date", "document type", "subject", "issuing authority") for c in cells):
                    continue
                
                # We are in table rows
                if len(cells) >= 2:
                    name = ""
                    role = ""
                    
                    # Check columns based on headers if available
                    name_idx = -1
                    role_idx = -1
                    for idx, h in enumerate(headers):
                        if any(kw in h for kw in ("member", "name", "faculty advisor", "coordinator", "designation")):
                            if name_idx == -1:
                                name_idx = idx
                        if any(kw in h for kw in ("role", "department", "designation")):
                            if idx != name_idx:
                                role_idx = idx
                                
                    if name_idx != -1 and name_idx < len(cells):
                        name = cells[name_idx]
                    else:
                        name = cells[0]
                        
                    if role_idx != -1 and role_idx < len(cells):
                        role = cells[role_idx]
                    elif len(cells) > 1:
                        role = cells[1]
                        
                    name = clean_admin_member_name(name)
                    if not name or name.lower() in ("member", "designation", "s. no.", "s.no.", "value", "field") or len(name) < 3:
                        continue
                        
                    # Skip numeric serial numbers
                    if re.match(r'^\d+$', name):
                        continue
                        
                    resolved_name = self.resolver.resolve(name)
                    
                    dept_key = current_dept if current_dept else "Institute"
                    node_key = f"committee_member:{normalize_name(committee_type)}:{normalize_name(dept_key)}:{normalize_name(resolved_name)}"
                    
                    self._add_node(node_key, "CommitteeMember",
                                   name=resolved_name,
                                   designation=role,
                                   department=dept_key,
                                   committee_type=committee_type,
                                   committee_name=title,
                                   notification_date=date_str,
                                   source_file=filename)
                    self._add_edge(node_key, doc_id, "SOURCE_DOCUMENT")
                    
                    # Connect to academics section node
                    academics_sec_node = f"section:{self.section_code}"
                    if self.graph.has_node(academics_sec_node):
                        self._add_edge(node_key, academics_sec_node, "BELONGS_TO_SECTION")

    def _parse_advisor_document(self, filename: str, content: str, doc_id: str, label: str):
        lines = content.splitlines()
        
        # Extract notification date if present
        date_match = re.search(r'\*\*Date:\*\*\s*([0-9a-zA-Z\s,]+)', content, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None
        
        # Extract batch year from filename/title (e.g. 2025)
        batch_year = "2025"
        year_match = re.search(r'\b(20\d{2})\b', filename)
        if year_match:
            batch_year = year_match.group(1)
            
        in_table = False
        headers = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('|'):
                cells = [c.strip() for c in line_stripped.split('|')[1:-1]]
                if not cells:
                    continue
                if all(re.match(r'^:?-+:?$', c) for c in cells):
                    in_table = True
                    continue
                if not in_table:
                    headers = [c.lower() for c in cells]
                    continue
                
                # Check for metadata
                if any(c.lower() in ("field", "value", "institution", "date", "document type") for c in cells):
                    continue
                
                # Table rows
                if len(cells) >= 3:
                    prog_name = cells[1]
                    name = clean_admin_member_name(cells[2])
                    
                    if not name or len(name) < 3:
                        continue
                    if re.match(r'^\d+$', name):
                        continue
                        
                    resolved_name = self.resolver.resolve(name)
                    node_key = f"{label.lower()}:{normalize_name(prog_name)}:{normalize_name(resolved_name)}"
                    
                    self._add_node(node_key, label,
                                   name=resolved_name,
                                   programme=prog_name,
                                   batch_year=batch_year,
                                   notification_date=date_str,
                                   source_file=filename)
                    self._add_edge(node_key, doc_id, "SOURCE_DOCUMENT")
                    
                    # Connect to academics section node
                    academics_sec_node = f"section:{self.section_code}"
                    if self.graph.has_node(academics_sec_node):
                        self._add_edge(node_key, academics_sec_node, "BELONGS_TO_SECTION")

    def _parse_fee_structure_document(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        
        # Extract notification date if present
        date_match = re.search(r'\*\*Date:\*\*\s*([0-9a-zA-Z\s,]+)', content, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None
        
        current_category = "General"
        in_table = False
        headers = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check headers
            category_match = re.match(r'^(?:#|##)\s+(.+?)(?:\s+Programmes|\s+Programmes\s*\(Continued\))?$', line_stripped, re.IGNORECASE)
            if category_match:
                match_val = category_match.group(1).strip()
                if "b.tech" in match_val.lower() or "ug-bs" in match_val.lower() or "m.tech" in match_val.lower() or "m.sc" in match_val.lower() or "ph.d" in match_val.lower():
                    current_category = match_val
                    in_table = False
                    continue
            
            if line_stripped.startswith('|'):
                cells = [c.strip() for c in line_stripped.split('|')[1:-1]]
                if not cells:
                    continue
                if all(re.match(r'^:?-+:?$', c) for c in cells):
                    in_table = True
                    continue
                if not in_table:
                    headers = [c.lower() for c in cells]
                    continue
                
                # Check for metadata
                if any(c.lower() in ("field", "value", "institution", "date", "document type") for c in cells):
                    continue
                
                # Table rows
                if len(cells) >= 3:
                    entry_year = cells[0]
                    # Skip if not starting with a year
                    if not re.search(r'\b\d{4}\b', entry_year):
                        continue
                        
                    income_category = "All"
                    if "income category" in headers:
                        income_idx = headers.index("income category")
                        income_category = cells[income_idx]
                    
                    programme = current_category
                    if "programme" in headers:
                        prog_idx = headers.index("programme")
                        programme = cells[prog_idx]
                    elif "category" in headers:
                        cat_idx = headers.index("category")
                        programme = f"{current_category} ({cells[cat_idx]})"
                        
                    # Find Fee Gen and Fee SC/ST
                    fee_gen_idx = -1
                    fee_sc_idx = -1
                    for idx, h in enumerate(headers):
                        if any(kw in h for kw in ("general/obc/ews", "general/obc/ews fee", "tuition fee + other charges")):
                            fee_gen_idx = idx
                        if any(kw in h for kw in ("sc/st/pwd", "sc/st/pwd fee", "sc/st/pwd charges")):
                            fee_sc_idx = idx
                            
                    fee_gen = ""
                    fee_sc_st_pwd = ""
                    if fee_gen_idx != -1 and fee_gen_idx < len(cells):
                        fee_gen = cells[fee_gen_idx]
                    elif len(cells) >= 3:
                        fee_gen = cells[2]
                        
                    if fee_sc_idx != -1 and fee_sc_idx < len(cells):
                        fee_sc_st_pwd = cells[fee_sc_idx]
                    elif len(cells) >= 4:
                        fee_sc_st_pwd = cells[3]
                        
                    # Create structured fee structure node
                    node_key = f"fee_structure:{normalize_name(programme)}:{normalize_name(entry_year)}:{normalize_name(income_category)}"
                    self._add_node(node_key, "FeeStructure",
                                   programme=programme,
                                   entry_year=entry_year,
                                   income_category=income_category,
                                   fee_gen_obc_ews=fee_gen,
                                   fee_sc_st_pwd=fee_sc_st_pwd,
                                   category=current_category,
                                   notification_date=date_str,
                                   source_file=filename)
                    self._add_edge(node_key, doc_id, "SOURCE_DOCUMENT")
                    
                    # Connect to academics section node
                    academics_sec_node = f"section:{self.section_code}"
                    if self.graph.has_node(academics_sec_node):
                        self._add_edge(node_key, academics_sec_node, "BELONGS_TO_SECTION")

    def _parse_notification_policy_document(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        
        # 1. Extract notification number
        notification_number = None
        number_match = re.search(r'(?:No\.|Notification No\.?:?)\s*(IITJMU/[^\s\n,\*;]+)', content, re.IGNORECASE)
        if number_match:
            notification_number = number_match.group(1).strip()
            notification_number = re.sub(r'[.\s]+$', '', notification_number)

        # 2. Extract notification date
        date_match = re.search(r'\*\*Date:\*\*\s*([0-9a-zA-Z\s,]+)', content, re.IGNORECASE)
        if not date_match:
            date_match = re.search(r'(?:Date|Dated)(?::-|:)?\s*([0-9]{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z,.\s]+\s+[0-9]{4})', content, re.IGNORECASE)
        if not date_match:
            date_match = re.search(r'(?:Date|Dated)(?::-|:)?\s*([0-9a-zA-Z\s,.-]+)', content, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else None
        if date_str:
            date_str = re.sub(r'\s+', ' ', date_str)

        # 3. Determine clean title from first H1 or filename
        title = ""
        h1_match = re.search(r'^\s*#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        else:
            title = os.path.basename(filename).replace(".md", "").replace("_", " ").title()
        title = re.sub(r'\*+', '', title).strip()

        # 4. Classify category based on filename and content
        fn_lower = filename.lower()
        content_lower = content.lower()
        
        category = "general"
        if any(w in fn_lower for w in ["internship", "six-month", "6-month"]):
            category = "internship_policy"
        elif any(w in fn_lower for w in ["fee_waiver", "tuition_fee"]):
            category = "financial_policy"
        elif any(w in fn_lower for w in ["transfer_of_doctoral", "phd_transfer"]):
            category = "phd_policy"
        elif any(w in fn_lower for w in ["grade_moderation", "moderation"]):
            category = "grading_policy"
        elif any(w in fn_lower for w in ["early_start_phd", "early_start"]):
            category = "phd_policy"
        elif any(w in fn_lower for w in ["phd_scholars_quota", "quota_under_project"]):
            category = "phd_policy"
        elif any(w in fn_lower for w in ["open_research_day", "open_research"]):
            category = "phd_policy"
        elif any(w in fn_lower for w in ["new_pg_program", "pg_programs", "starting_new_pg", "introducing_new_m"]):
            category = "pg_procedure"
        elif any(w in fn_lower for w in ["backlog", "re-examination", "re_examination"]):
            category = "grading_policy"
        elif any(w in fn_lower for w in ["partial_financial_support", "international_students"]):
            category = "financial_policy"
        elif any(w in fn_lower for w in ["study_in_india", "sii"]):
            category = "admission_policy"
        elif any(w in fn_lower for w in ["foreign_nationals", "foreign_national"]):
            category = "admission_policy"
        elif any(w in fn_lower for w in ["spoc", "spoc_notification"]):
            category = "admin_notification"
        elif any(w in fn_lower for w in ["stic_dinner", "stic"]):
            category = "admin_notification"
        elif any(w in fn_lower for w in ["fellowship_extension", "htra_to_female", "extension_of_htra"]):
            category = "phd_policy"
        elif any(w in fn_lower for w in ["revision_of_hra", "hra_on_fellowship"]):
            category = "financial_policy"
        elif any(w in fn_lower for w in ["summer_term_financial", "financial_incentive"]):
            category = "financial_policy"

        # 5. Extract applies_to
        applies_to = []
        if "b.tech" in content_lower or "ug" in content_lower:
            applies_to.append("B.Tech")
        if "m.tech" in content_lower or "pg" in content_lower or "m.sc" in content_lower or "ms(r)" in content_lower:
            applies_to.append("PG")
        if "phd" in content_lower or "ph.d" in content_lower or "doctoral" in content_lower:
            applies_to.append("PhD")
        if "international" in content_lower or "foreign" in content_lower:
            applies_to.append("International Students")
        if not applies_to:
            applies_to = ["All"]

        # 6. Keywords
        keywords = []
        kw_options = ["internship", "fee waiver", "remission", "phd transfer", "backlog", "re-examination", "grade moderation",
                      "early start", "project funding", "open research day", "pg programs", "stipend", "fellowship", "admission",
                      "hra", "contingency", "credit limit", "preparatory", "mini project", "study in india", "spoc", "stic dinner"]
        for kw in kw_options:
            if kw in content_lower:
                keywords.append(kw.title())

        # 7. Extract Summary (first few paragraphs)
        summary_paragraphs = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#") or line_str.startswith("---") or line_str.startswith("<!--") or line_str.startswith("**") or ":" in line_str[:20]:
                continue
            summary_paragraphs.append(line_str)
            if len(summary_paragraphs) >= 3:
                break
        summary = "\n\n".join(summary_paragraphs)

        # 8. Eligibility Criteria extraction
        eligibility_criteria = []
        in_eligibility_section = False
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("#") and any(w in line_str.lower() for w in ["eligibility", "condition", "requirement", "guideline"]):
                in_eligibility_section = True
                continue
            elif line_str.startswith("#"):
                in_eligibility_section = False
                
            if in_eligibility_section:
                if line_str.startswith("-") or line_str.startswith("*"):
                    item = line_str.lstrip("-* ").strip()
                    if item: eligibility_criteria.append(item)
                elif re.match(r'^\d+\.\s*', line_str):
                    item = re.sub(r'^\d+\.\s*', '', line_str).strip()
                    if item: eligibility_criteria.append(item)

        # 9. Procedure Steps extraction
        procedure_steps = []
        in_procedure_section = False
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("#") and any(w in line_str.lower() for w in ["procedure", "step", "process", "workflow", "how to"]):
                in_procedure_section = True
                continue
            elif line_str.startswith("#"):
                in_procedure_section = False
                
            if in_procedure_section:
                if line_str.startswith("-") or line_str.startswith("*"):
                    item = line_str.lstrip("-* ").strip()
                    if item: procedure_steps.append(item)
                elif re.match(r'^\d+\.\s*', line_str) or line_str.startswith(">"):
                    item = re.sub(r'^\d+\.\s*|>\s*', '', line_str).strip()
                    if item: procedure_steps.append(item)

        # 10. Extract key facts (slabs, deadlines, limits, numbers)
        key_facts = []
        if notification_number:
            key_facts.append({"key": "Notification Number", "value": notification_number})
        if date_str:
            key_facts.append({"key": "Notification Date", "value": date_str})

        if "tuition_fee_waiver" in fn_lower or "fee_waiver" in fn_lower or "tuition" in fn_lower:
            key_facts.extend([
                {"key": "SC / ST / PH Students Benefit", "value": "100% Tuition Fee Waiver"},
                {"key": "Most Economically Backward (Income < 1 Lakh) Benefit", "value": "100% Tuition Fee Remission"},
                {"key": "Other Economically Backward (Income 1 Lakh to 5 Lakh) Benefit", "value": "2/3rd Tuition Fee Remission"},
                {"key": "Income Above 5 Lakh Benefit", "value": "No Tuition Fee Remission"},
                {"key": "Submission Portal", "value": "SARAL Portal"},
                {"key": "Portal Request Type", "value": "Income Certificate / Category Change Request"},
                {"key": "Submission Deadline for AY 2026-27", "value": "15 June 2026"}
            ])
        elif "six-month_internship" in fn_lower or "internship_policy" in fn_lower or "internship" in fn_lower:
            key_facts.extend([
                {"key": "Internship Duration", "value": "Up to 6 months"},
                {"key": "B.Tech Eligibility Window", "value": "8th Semester (December - June)"},
                {"key": "M.Tech/M.Sc. Eligibility Window", "value": "4th Semester (December - June)"},
                {"key": "M.Tech/MS(R) (3 Years) Eligibility Window", "value": "6th Semester (December - June)"},
                {"key": "UG Course Allowance During Internship", "value": "Maximum 1 online course (with prior approval)"},
                {"key": "PG Course Allowance During Internship", "value": "No online courses allowed. Must complete coursework first."},
                {"key": "Stipend/Scholarship", "value": "No stipend or scholarship provided by the Institute during internship"},
                {"key": "Placements Restriction", "value": "If a student has a confirmed 6-month internship offer, they are not eligible for other placement/I+J opportunities"}
            ])
        elif "transfer_of_doctoral" in fn_lower or "transfer" in fn_lower:
            key_facts.extend([
                {"key": "Eligible Sending Institutions", "value": "IISc, IITs, NITs, IISERs, ISI, IMSc, CMI, NIRF Top 100 relevant discipline, Foreign QS < 750"},
                {"key": "Maximum Prior Registration at Sending Institute", "value": "2 Years"},
                {"key": "Minimum CGPA for Transfer", "value": "8.0"},
                {"key": "Transfer Timeline", "value": "Must start within one year of the supervisor joining IIT Jammu"},
                {"key": "Minimum Residency Requirement", "value": "One Year enrollment at IIT Jammu before thesis submission"},
                {"key": "Maximum Scholarship Support Duration", "value": "5 Years (including support received at transferring institution)"}
            ])
        elif "scholars_quota" in fn_lower or "project_funding" in fn_lower:
            key_facts.extend([
                {"key": "Additional PhD Scholar Allowance", "value": "One additional PhD scholar under Project Funding beyond normal quota"},
                {"key": "Minimum Service Requirement for Faculty", "value": "Five years of service at IIT Jammu"},
                {"key": "PhD Graduation Record Requirement", "value": "At least one PhD scholar defended at IIT Jammu"},
                {"key": "Remaining Project Funding Duration", "value": "Minimum 3 years remaining"},
                {"key": "Conversion Limit to Institute Funding", "value": "Maximum 2 project-funded PhD students under the faculty member"}
            ])
        elif "grade_moderation" in fn_lower or "moderation" in fn_lower:
            key_facts.extend([
                {"key": "Maximum AA Grade Percentage", "value": "10% of students in any course"},
                {"key": "Minimum Passing Marks", "value": "Above 30% on an absolute scale"},
                {"key": "Expected Course Average Grade Range", "value": "6.5 to 7.5"},
                {"key": "Maximum Allowed Average Grade", "value": "8.50"},
                {"key": "Normal Grade Distribution Requirement", "value": "Expected in courses with more than 30 graded students"}
            ])
        elif "early_start" in fn_lower:
            key_facts.extend([
                {"key": "GATE Requirement", "value": "No GATE qualification is required"},
                {"key": "Minimum UG CGPA/CPI for General/OBC/EWS", "value": "8.0 at the end of the third year"},
                {"key": "Minimum UG CGPA/CPI for SC/ST/PwD", "value": "7.5 at the end of the third year"},
                {"key": "Department Quota", "value": "One early start PhD position per department per year"},
                {"key": "Maximum Fellowship Duration", "value": "5 years"},
                {"key": "Fee Payment Deadline", "value": "Within two weeks from the date of email notification"}
            ])

        amount_matches = re.findall(r'(?:Rs\.?|INR|₹)\s*[\d,]+', content)
        for amt in amount_matches[:3]:
            context_match = re.search(r'([^.\n]{0,30}' + re.escape(amt) + r'[^.\n]{0,30})', content)
            if context_match:
                key_facts.append({"key": "Financial Detail", "value": context_match.group(1).strip()})

        # Create structured notification node
        node_key = f"policy_notification:{normalize_name(title)}"
        self._add_node(node_key, "PolicyNotification",
                       title=title,
                       notification_number=notification_number,
                       notification_date=date_str,
                       category=category,
                       applies_to=applies_to,
                       keywords=keywords,
                       summary=summary,
                       key_facts=key_facts,
                       eligibility_criteria=eligibility_criteria,
                       procedure_steps=procedure_steps,
                       source_file=filename)
                       
        self._add_edge(node_key, doc_id, "SOURCE_DOCUMENT")
        
        # Connect to academics section node
        academics_sec_node = f"section:{self.section_code}"
        if self.graph.has_node(academics_sec_node):
            self._add_edge(node_key, academics_sec_node, "BELONGS_TO_SECTION")

    def _parse_curriculum_document(self, filename: str, content: str, doc_id: str):
        """Parse academic curriculum and specialization documents."""
        # 1. Determine level (UG/PG) and category/type from filename
        level = "UG"
        if "pg_mtech" in filename:
            level = "PG (M.Tech)"
        elif "pg_msc" in filename:
            level = "PG (M.Sc)"
        elif "pg_phd" in filename:
            level = "PG (PhD)"

        prog_type = "Course Offering Framework"
        if "specialization" in filename:
            if "micro" in filename:
                prog_type = "Micro Specialization"
            elif "minor" in filename:
                prog_type = "Minor"
            elif "honours" in filename or "honor" in filename:
                prog_type = "Honours"
            else:
                prog_type = "Specialization"

        # 2. Extract program or specialization title from the first H1 header
        title = ""
        first_h1_match = re.search(r'^\s*#\s+(.+)$', content, re.MULTILINE)
        if first_h1_match:
            title = first_h1_match.group(1).strip()
        else:
            title = filename.replace("academics_curriculum_", "").replace(".md", "").replace("_", " ").title()

        # Clean title
        title = re.sub(r'\*+', '', title).strip()

        # Handle superseded versions (check for keywords in filename/content)
        superseded = False
        if any(k in filename.lower() for k in ("before-2024", "before_2024", "2019_scheme", "2019-scheme", "2017", "2018")):
            superseded = True

        # Extract total credits / graduation requirement if present in content
        total_credits = None
        req_match = re.search(r'graduation\s+requirement[s]?\s*[:*]*\s*([0-9.]+)(?:\s*credits)?', content, re.IGNORECASE)
        if req_match:
            total_credits = req_match.group(1).strip()
        else:
            tot_match = re.search(r'total\s+(?:program\s+)?credits\s*[:*]*\s*([0-9./-]+)', content, re.IGNORECASE)
            if tot_match:
                total_credits = tot_match.group(1).strip()

        # 3. Create Program or Specialization Node
        dept = self._infer_department(title)
        if prog_type in ("Minor", "Specialization", "Micro Specialization", "Honours"):
            entity_label = "Specialization"
            entity_id = f"specialization:{normalize_name(title)}"
            self._add_node(entity_id, "Specialization",
                           name=title,
                           type=prog_type,
                           level=level,
                           department=dept,
                           superseded=superseded,
                           total_credits=total_credits,
                           source_file=filename)
        else:
            entity_label = "AcademicProgram"
            entity_id = f"program:{normalize_name(title)}"
            self._add_node(entity_id, "AcademicProgram",
                           name=title,
                           level=level,
                           department=dept,
                           superseded=superseded,
                           total_credits=total_credits,
                           source_file=filename)

        self._add_edge(entity_id, doc_id, "SOURCE_DOCUMENT")

        # Connect Academics section node to the Program/Specialization
        academics_sec_node = f"section:{self.section_code}"
        if self.graph.has_node(academics_sec_node):
            self._add_edge(academics_sec_node, entity_id, "OFFERS_PROGRAM" if entity_label == "AcademicProgram" else "OFFERS_SPECIALIZATION")

        # 4. Parse Tables
        lines = content.splitlines()
        table_lines = []
        in_table = False
        current_semester = None
        current_category = None
        current_bucket = None

        # Helper to parse standard table
        def parse_table(t_lines):
            if len(t_lines) < 3:
                return []
            
            # Let's search for the header row among the rows of the table
            # Usually it's row 0, but could be row 2 if row 0 is a title row like "Semester-I"
            header_row_idx = 0
            header_cells = [c.strip() for c in t_lines[0].split('|')[1:-1]]
            
            def check_headers(cells):
                name_found = False
                code_found = False
                for cell in cells:
                    cell_lower = cell.lower()
                    if any(k in cell_lower for k in ("course name", "coursename", "course_name", "subject", "subject name", "subject_name", "subjectname", "name of the course", "course title", "course_title", "coursetitle", "course")):
                        name_found = True
                    if any(k in cell_lower for k in ("course no.", "course no", "course code", "subject code", "code", "course_code", "coursecode", "subjectcode")):
                        code_found = True
                return name_found or code_found

            if not check_headers(header_cells):
                # Try row 2
                if len(t_lines) > 2:
                    row2_cells = [c.strip() for c in t_lines[2].split('|')[1:-1]]
                    if check_headers(row2_cells):
                        header_cells = row2_cells
                        header_row_idx = 2

            name_idx = -1
            code_idx = -1
            ltp_idx = -1
            credits_idx = -1
            l_idx, t_idx, p_idx = -1, -1, -1
            has_course_header = False
            for idx, cell in enumerate(header_cells):
                cell_lower = cell.lower()
                if any(k in cell_lower for k in ("course name", "coursename", "course_name", "subject", "subject name", "subject_name", "subjectname", "name of the course", "course title", "course_title", "coursetitle")) and name_idx == -1:
                    name_idx = idx
                    has_course_header = True
                elif cell_lower == "course" and name_idx == -1:
                    name_idx = idx
                    has_course_header = True
                elif any(k in cell_lower for k in ("course no.", "course no", "course code", "subject code", "code", "course_code", "coursecode", "subjectcode")) and code_idx == -1:
                    code_idx = idx
                    has_course_header = True
                elif any(k in cell_lower for k in ("l-t-p", "l-t-p-c", "credit structure", "ltp", "structure")) and ltp_idx == -1:
                    ltp_idx = idx
                elif any(k in cell_lower for k in ("total credits", "totalcredit", "totalcredits", "credits", "credit", "c")) and credits_idx == -1:
                    if cell_lower != "co-curricular":
                        credits_idx = idx
                elif cell_lower == "l":
                    l_idx = idx
                elif cell_lower == "t":
                    t_idx = idx
                elif cell_lower == "p":
                    p_idx = idx

            if not has_course_header:
                return []

            results = []
            start_row = max(2, header_row_idx + 1)
            for line in t_lines[start_row:]:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if not cells or len(cells) < max(name_idx, code_idx, ltp_idx, credits_idx, l_idx, t_idx, p_idx) + 1:
                    continue

                course_name = cells[name_idx] if name_idx != -1 else ""
                course_code = cells[code_idx] if code_idx != -1 else ""
                ltp = cells[ltp_idx] if ltp_idx != -1 else ""
                credits_val = cells[credits_idx] if credits_idx != -1 else ""

                if not course_name or course_name.lower() in ("total", "semester credits", "course", "subject", "course name"):
                    continue
                if re.search(r'^\s*total\b', course_name.lower()) or re.search(r'semester credits', course_name.lower()):
                    continue

                course_name = re.sub(r'\*+', '', course_name).strip()
                course_name = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', course_name).strip()

                if any(placeholder in course_name.lower() for placeholder in ("elective-", "elective -", "dept. elective", "open elective", "department elective", "hss-", "hss i", "hss ii", "hss elective")):
                    continue

                if l_idx != -1 and t_idx != -1 and p_idx != -1:
                    l_val = cells[l_idx]
                    t_val = cells[t_idx]
                    p_val = cells[p_idx]
                    ltp = f"{l_val}-{t_val}-{p_val}"

                if ltp and ":" in ltp:
                    parts = ltp.split(":")
                    ltp = parts[0].strip()
                    if not credits_val:
                        credits_val = parts[1].strip()

                course_code = re.sub(r'\s+', '', course_code).strip()

                results.append({
                    "name": course_name,
                    "code": course_code,
                    "ltp": ltp,
                    "credits": credits_val
                })
            return results

        # Helper to parse key-value table
        def parse_kv_table(t_lines):
            c_name, c_code, c_ltp, c_credits = "", "", "", ""
            for line in t_lines:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) < 3:
                    continue
                key = cells[1].strip().lower()
                val = cells[2].strip()

                if "course title" in key or "course_title" in key or key == "course":
                    c_name = val
                elif "course number" in key or "course_number" in key or "course code" in key or key in ("course no", "course no."):
                    c_code = val
                elif "l-t-p" in key or "ltp structure" in key or "ltp_structure" in key or key == "structure":
                    c_ltp = val
                elif "credits" in key or key == "credit":
                    c_credits = val

            if c_name and (c_code or c_ltp or c_credits):
                c_name = re.sub(r'\*+', '', c_name).strip()
                c_code = re.sub(r'\s+', '', c_code).strip()
                return {
                    "name": c_name,
                    "code": c_code,
                    "ltp": c_ltp,
                    "credits": c_credits
                }
            return None

        # 5. Process Table Lines & Heading Contexts
        for idx, line in enumerate(lines):
            line_stripped = line.strip()

            # Track semesters, categories, and buckets
            if line_stripped.startswith("## "):
                heading = line_stripped[3:].strip()
                # Check for semester
                sem_match = re.search(r'Semester\s+([IVXLCDM\d]+)', heading, re.IGNORECASE)
                if sem_match:
                    current_semester = sem_match.group(1).upper()
                else:
                    # Check for elective bucket or theme
                    if any(k in heading.lower() for k in ("elective", "basket", "track", "theme")):
                        current_bucket = heading
                        current_semester = None
                    else:
                        current_category = heading
                        current_semester = None
                        current_bucket = None
            elif line_stripped.startswith("### "):
                heading = line_stripped[4:].strip()
                if any(k in heading.lower() for k in ("elective", "basket", "track", "theme")):
                    current_bucket = heading
                else:
                    current_category = heading

            if line_stripped.startswith("|"):
                in_table = True
                table_lines.append(line_stripped)
            else:
                if in_table:
                    # Parse the table we just finished gathering
                    parsed_courses = parse_table(table_lines)
                    if not parsed_courses:
                        kv_c = parse_kv_table(table_lines)
                        if kv_c:
                            parsed_courses = [kv_c]

                    # Add parsed courses as nodes and connect to the program/specialization
                    for course in parsed_courses:
                        c_name = course["name"]
                        c_code = course["code"]
                        c_ltp = course["ltp"]
                        c_credits = course["credits"]

                        c_node_id = f"course:{normalize_name(c_code if c_code else c_name)}"
                        
                        self._add_node(c_node_id, "Course",
                                       name=c_name,
                                       code=c_code,
                                       ltp=c_ltp,
                                       credits=c_credits,
                                       source_file=filename)

                        self._add_edge(entity_id, c_node_id, "OFFERS_COURSE",
                                       semester=current_semester,
                                       category=current_category,
                                       bucket=current_bucket)

                        if current_bucket:
                            bucket_id = f"bucket:{normalize_name(current_bucket)}"
                            self._add_node(bucket_id, "ElectiveBucket",
                                           name=current_bucket,
                                           source_file=filename)
                            self._add_edge(c_node_id, bucket_id, "BELONGS_TO_BUCKET")
                            self._add_edge(entity_id, bucket_id, "HAS_BUCKET")

                    table_lines = []
                    in_table = False

        if in_table:
            # Handle trailing table at EOF
            parsed_courses = parse_table(table_lines)
            if not parsed_courses:
                kv_c = parse_kv_table(table_lines)
                if kv_c:
                    parsed_courses = [kv_c]
            for course in parsed_courses:
                c_name = course["name"]
                c_code = course["code"]
                c_ltp = course["ltp"]
                c_credits = course["credits"]
                c_node_id = f"course:{normalize_name(c_code if c_code else c_name)}"
                self._add_node(c_node_id, "Course",
                               name=c_name,
                               code=c_code,
                               ltp=c_ltp,
                               credits=c_credits,
                               source_file=filename)
                self._add_edge(entity_id, c_node_id, "OFFERS_COURSE",
                               semester=current_semester,
                               category=current_category,
                               bucket=current_bucket)
                if current_bucket:
                    bucket_id = f"bucket:{normalize_name(current_bucket)}"
                    self._add_node(bucket_id, "ElectiveBucket",
                                   name=current_bucket,
                                   source_file=filename)
                    self._add_edge(c_node_id, bucket_id, "BELONGS_TO_BUCKET")
                    self._add_edge(entity_id, bucket_id, "HAS_BUCKET")

    def build(self) -> nx.DiGraph:
        if not os.path.exists(self.markdown_dir):
            raise FileNotFoundError(f"Markdown directory not found: {self.markdown_dir}")

        filenames = [f for f in os.listdir(self.markdown_dir)
                     if f.endswith(".md") and not f.startswith("00_combined")
                     and not f.endswith(".json")]

        # Discover files in subdirectories for academics section (academic_notifications)
        if self.section_code == "academics":
            notifications_dir = os.path.join(self.markdown_dir, "parsed_documents", "academic_notifications")
            if os.path.exists(notifications_dir):
                notification_files = [f for f in os.listdir(notifications_dir)
                                      if f.endswith(".md") and not f.startswith("00_combined")
                                      and not f.endswith(".json")]
                for nf in notification_files:
                    # Keep relative path under markdown_dir
                    filenames.append(os.path.join("parsed_documents", "academic_notifications", nf))

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
                elif self.section_code == "osd":
                    self._parse_osd_team(filename, content, doc_id)
                elif self.section_code == "ir":
                    self._parse_ir_team(filename, content, doc_id)
                else:
                    self._parse_people_list(filename, content, doc_id)
            elif "know-your-counselors" in filename:
                self._parse_counselor_profiles(filename, content, doc_id)
            elif "hod-message" in filename:
                self._parse_hod_message(filename, content, doc_id)
            elif "contact" in filename:
                if self.section_code == "alumni-affairs":
                    self._parse_alumni_contacts(filename, content, doc_id)
                elif self.section_code == "cds":
                    self._parse_cds_contact(filename, content, doc_id)
                elif self.section_code == "ir":
                    self._parse_ir_contact(filename, content, doc_id)
                elif self.section_code == "medical-centre":
                    self._parse_medical_contact(filename, content, doc_id)
                elif self.section_code == "osd":
                    self._parse_osd_contact(filename, content, doc_id)
                else:
                    self._parse_section_contact(filename, content, doc_id)
            
            # Custom Section-Specific Dispatch Rules
            if self.section_code == "alumni-affairs":
                if "medalist" in filename:
                    self._parse_alumni_medalists(filename, content, doc_id)
                elif "award" in filename:
                    self._parse_alumni_awards(filename, content, doc_id)
            elif self.section_code == "cds":
                if "past-recruiters" in filename:
                    self._parse_past_recruiters(filename, content, doc_id)
                elif "Placement_Policy" in filename:
                    self._parse_placement_policy(filename, content, doc_id)
                elif "aipc" in filename:
                    self._parse_aipc_guidelines(filename, content, doc_id)
                elif "rise-up" in filename or "RISE-UP" in filename or "RISE_UP" in filename:
                    self._parse_rise_up_details(filename, content, doc_id)
                elif "Placement_Report" in filename:
                    self._parse_placement_stats(filename, content, doc_id)
            elif self.section_code == "ir":
                if "mous" in filename:
                    self._parse_mous(filename, content, doc_id)
                elif "clubs" in filename:
                    self._parse_clubs(filename, content, doc_id)
                elif "sports" in filename:
                    self._parse_sports(filename, content, doc_id)
                elif "residential" in filename:
                    self._parse_hostels(filename, content, doc_id)
                elif "fests" in filename:
                    self._parse_fests(filename, content, doc_id)
            elif self.section_code == "medical-centre":
                if "about" in filename:
                    self._parse_medical_about(filename, content, doc_id)
                elif "know-your-doctors" in filename or "doctors" in filename:
                    self._parse_medical_doctors(filename, content, doc_id)
                elif "collaborations" in filename:
                    self._parse_medical_collaborations(filename, content, doc_id)
                elif any(svc in filename for svc in ("dental", "ward", "dressing", "physiotherapy", "laboratory", "ecg", "pharmacy", "ambulance")):
                    self._parse_medical_services(filename, content, doc_id)
            elif self.section_code == "osd":
                if "unnat-bharat" in filename:
                    self._parse_osd_uba(filename, content, doc_id)
                elif "ces" in filename:
                    self._parse_osd_ces(filename, content, doc_id)
                elif "events" in filename:
                    self._parse_osd_events(filename, content, doc_id)
            elif self.section_code == "academics":
                if "curriculum" in filename:
                    self._parse_curriculum_document(filename, content, doc_id)
                elif "specialisation-and-courses" in filename:
                    self._parse_specialisation_index(filename, content, doc_id)
                elif "DPGC" in filename or "DUGC" in filename or "Committee" in filename:
                    self._parse_committee_document(filename, content, doc_id)
                elif "Faculty_Advisor" in filename:
                    self._parse_advisor_document(filename, content, doc_id, "FacultyAdvisor")
                elif "Program_Coordinator" in filename:
                    self._parse_advisor_document(filename, content, doc_id, "ProgramCoordinator")
                elif "Fee_Notification" in filename:
                    self._parse_fee_structure_document(filename, content, doc_id)
                elif "academic_notifications" in filename:
                    self._parse_notification_policy_document(filename, content, doc_id)

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

    def _infer_department(self, title_str: str) -> Optional[str]:
        title_lower = title_str.lower()
        
        def has_word(w: str) -> bool:
            return bool(re.search(r'\b' + re.escape(w) + r'\b', title_lower))

        if "computer science" in title_lower or has_word("cse") or "data science" in title_lower or "information security" in title_lower:
            return "computer_science_engineering"
        if "electrical" in title_lower or has_word("ee") or "vlsi" in title_lower or "microelectronics" in title_lower or "communication and signal" in title_lower or "cyber physical systems" in title_lower:
            return "ee"
        if "civil" in title_lower or "geotechnical" in title_lower or "structural" in title_lower or "tunnel" in title_lower or has_word("ce"):
            return "civil_engineering"
        if "mechanical" in title_lower or "thermal" in title_lower or "system design" in title_lower or "energy systems" in title_lower or has_word("me"):
            return "mechanical_engineering"
        if "chemical" in title_lower or "sustainable energy" in title_lower or has_word("ch") or has_word("che"):
            return "chemical-engineering"
        if "bioengineering" in title_lower or "biosciences" in title_lower or has_word("bsbe"):
            return "bsbe"
        if "chemistry" in title_lower:
            return "chemistry"
        if "physics" in title_lower:
            return "physics"
        if "materials" in title_lower or "metallurgy" in title_lower or has_word("mt") or has_word("mty"):
            return "materials-engineering"
        if "mathematics" in title_lower or "computing" in title_lower:
            return "mathematics"
        if "economics" in title_lower or has_word("hss"):
            return "hss"
        return None

    def _parse_specialisation_index(self, filename: str, content: str, doc_id: str):
        """Parse the academics-specialisation-and-courses index file."""
        lines = content.splitlines()
        current_level = "UG"
        current_type = ""

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if line_stripped.startswith("## "):
                heading = line_stripped[3:].strip()
                if "PG" in heading.upper():
                    current_level = "PG"
                elif "UG" in heading.upper():
                    current_level = "UG"
                continue

            subheading_match = re.match(r'^\d+\.\s+([A-Za-z.\s&/_()\-]+)$', line_stripped)
            if subheading_match:
                current_type = subheading_match.group(1).strip()
                continue

            link_match = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line_stripped)
            for link_text, link_url in link_match:
                title = re.sub(r'\*+', '', link_text).strip()
                superseded = False
                if any(k in title.lower() or k in link_url.lower() for k in ("before-2024", "before_2024", "2019_scheme", "2019-scheme", "2017", "2018", "old")):
                    superseded = True

                prog_type = current_type
                if "micro" in title.lower():
                    prog_type = "Micro Specialization"
                elif "minor" in title.lower():
                    prog_type = "Minor"
                elif "honours" in title.lower() or "honor" in title.lower():
                    prog_type = "Honours"

                # Extract total credits / graduation requirement if present in content
                total_credits = None
                req_match = re.search(r'graduation\s+requirement[s]?\s*[:*]*\s*([0-9.]+)(?:\s*credits)?', content, re.IGNORECASE)
                if req_match:
                    total_credits = req_match.group(1).strip()
                else:
                    tot_match = re.search(r'total\s+(?:program\s+)?credits\s*[:*]*\s*([0-9./-]+)', content, re.IGNORECASE)
                    if tot_match:
                        total_credits = tot_match.group(1).strip()

                dept = self._infer_department(title)

                if prog_type in ("Minor", "Specialization", "Micro Specialization", "Honours") or "specialization" in title.lower() or "minor" in title.lower() or "micro" in title.lower() or "honours" in title.lower():
                    entity_label = "Specialization"
                    entity_id = f"specialization:{normalize_name(title)}"
                    self._add_node(entity_id, "Specialization",
                                   name=title,
                                   type=prog_type if prog_type else "Specialization",
                                   level=current_level,
                                   link=link_url,
                                   department=dept,
                                   superseded=superseded,
                                   total_credits=total_credits,
                                   source_file=filename)
                else:
                    entity_label = "AcademicProgram"
                    entity_id = f"program:{normalize_name(title)}"
                    self._add_node(entity_id, "AcademicProgram",
                                   name=title,
                                   level=f"{current_level} ({prog_type})" if prog_type else current_level,
                                   link=link_url,
                                   department=dept,
                                   superseded=superseded,
                                   total_credits=total_credits,
                                   source_file=filename)

                self._add_edge(entity_id, doc_id, "SOURCE_DOCUMENT")

                academics_sec_node = f"section:{self.section_code}"
                if self.graph.has_node(academics_sec_node):
                    self._add_edge(academics_sec_node, entity_id, "OFFERS_PROGRAM" if entity_label == "AcademicProgram" else "OFFERS_SPECIALIZATION")

    def _parse_alumni_medalists(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        
        # 1. First pass: extract all available years per program from the links
        program_years = {
            "B.Tech": [],
            "M.Tech": [],
            "Ph.D.": [],
            "M.Sc.": []
        }
        for line in lines:
            prog_match = re.search(r'\[(B\.Tech|M\.Tech|Ph\.D\.|M\.Sc\.)\s*\((\d{4})\)\]', line)
            if prog_match:
                prog = prog_match.group(1)
                yr = int(prog_match.group(2))
                if yr not in program_years[prog]:
                    program_years[prog].append(yr)
        
        # Sort the years just to be sure
        for prog in program_years:
            program_years[prog].sort()
            
        # Fallback values if none were found
        if not program_years["B.Tech"]:
            program_years["B.Tech"] = [2020, 2021, 2022, 2023, 2024]
        if not program_years["M.Tech"]:
            program_years["M.Tech"] = [2021, 2022, 2023, 2024]
        if not program_years["Ph.D."]:
            program_years["Ph.D."] = [2022, 2023]
        if not program_years["M.Sc."]:
            program_years["M.Sc."] = [2024]
            
        # 2. Second pass: parse the medalists
        current_program = "B.Tech"
        year_index = 0
        current_award = ""
        
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check for program link to switch context
            prog_match = re.search(r'\[(B\.Tech|M\.Tech|Ph\.D\.|M\.Sc\.)\s*\((\d{4})\)\]', line)
            if prog_match:
                new_program = prog_match.group(1)
                if new_program != current_program:
                    current_program = new_program
                    year_index = 0
            elif "Last Updated on" in line:
                years_list = program_years[current_program]
                year_index = min(year_index + 1, len(years_list) - 1)
            elif line.startswith("- [") and "Last Updated" not in line and not any(p in line for p in ("B.Tech", "M.Tech", "Ph.D.", "M.Sc.")):
                match = re.match(r'-\s*\[(.*?)\]', line)
                if match:
                    current_award = match.group(1).strip()
            elif line.startswith("#### "):
                name = line[5:].strip()
                dept = ""
                j = i + 1
                while j < len(lines) and j < i + 10:
                    if "Department" in lines[j]:
                        if j + 1 < len(lines) and lines[j+1].strip() and not lines[j+1].startswith("#") and not lines[j+1].startswith("-"):
                            dept = lines[j+1].strip()
                        elif j + 2 < len(lines) and lines[j+2].strip() and not lines[j+2].startswith("#") and not lines[j+2].startswith("-"):
                            dept = lines[j+2].strip()
                        break
                    j += 1
                
                if name and current_award:
                    resolved = self.resolver.resolve(name)
                    years_list = program_years[current_program]
                    year_val = years_list[year_index]
                    
                    node_id = f"medalist:{resolved.replace(' ', '_').lower()}:{year_val}"
                    self._add_node(node_id, "AlumniMedalist",
                                   name=resolved,
                                   award=current_award,
                                   year=year_val,
                                   degree=current_program,
                                   department=dept,
                                   source_file=filename)
                    self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")
            i += 1

    def _parse_alumni_awards(self, filename: str, content: str, doc_id: str):
        year = 2022
        year_match = re.search(r'award-(\d{4})', filename)
        if year_match:
            year = int(year_match.group(1))
            
        name = ""
        deg = ""
        dept = ""
        
        name_match = re.search(r'\*\*(?:Mr\.|Ms\.)\s*([A-Za-z\s\.]+?)(?:\s*\(|\*\*|\n)', content)
        if name_match:
            name = name_match.group(1).strip()
            
        if not name:
            name_match = re.search(r'(Harshdev Singh|Lakshya Devani|Pranav Reddy)', content)
            if name_match:
                name = name_match.group(1).strip()
                
        if "Electrical Engineering" in content:
            dept = "Electrical Engineering"
        if "B.Tech" in content:
            deg = "B.Tech"
            
        if name:
            resolved_name = self.resolver.resolve(name)
            award_id = f"award:{resolved_name}:{year}"
            self._add_node(award_id, "AlumniAward",
                           name=resolved_name,
                           award_name="Best New Initiative Award",
                           year=year,
                           degree=deg,
                           department=dept,
                           source_file=filename)
            self._add_edge(award_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_alumni_contacts(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#### "):
                name = line[5:].strip()
                designation = ""
                email = ""
                
                j = i + 1
                while j < len(lines) and not lines[j].startswith("#### "):
                    sub_line = lines[j]
                    if sub_line.startswith("##### "):
                        designation = sub_line[6:].strip()
                    elif "@" in sub_line:
                        email = _deobfuscate_email_text(sub_line)
                    j += 1
                    
                resolved_name = self.resolver.resolve(name)
                label = "SectionHead" if "Dean" in designation or "HoS" in designation else "SectionPerson"
                self._add_node(resolved_name, label,
                               name=resolved_name,
                               designation=designation,
                               email=email,
                               source_file=filename)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                i = j - 1
            i += 1

    # Custom Parsers for CDS
    def _parse_past_recruiters(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        for line in lines:
            if line.startswith("- ") and len(line) > 2:
                company = line[2:].strip()
                if "Image" not in company and "Recruiters" not in company:
                    node_id = f"recruiter:{company}"
                    self._add_node(node_id, "Recruiter",
                                   name=company,
                                   source_file=filename)
                    self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_placement_policy(self, filename: str, content: str, doc_id: str):
        rules = [
            ("CGPA Cutoff", "Minimum of 6 CGPA is required for student eligibility in placements.", "Eligibility & Registration"),
            ("Attendance Rule", "Attendance below 80% in guest lectures and workshops will result in being debarred from the first 5 companies.", "Attendance Rules"),
            ("Pre-Placement Talk Attendance", "If a student fails to attend any pre-placement talks/presentations/workshops by a company they applied/pre-registered for, they will not be allowed for that company's process.", "Attendance Rules"),
            ("Core vs Non-Core Offers", "A student can avail one offer in a core job category regardless of CTC. Core placed students can take one more offer in Non-Core (Category III or IV). Non-Core placed students cannot apply to Core.", "Application Rules"),
            ("CTC Category I", "Up to Rs. 8 lakhs per annum CTC", "CTC Criteria"),
            ("CTC Category II", "Rs. 8.01 up to Rs. 15 Lakhs per annum CTC", "CTC Criteria"),
            ("CTC Category III", "Rs. 15.01 up to Rs. 25 Lakhs per annum CTC", "CTC Criteria"),
            ("CTC Category IV", "Rs 25 lakhs and above per annum CTC", "CTC Criteria"),
            ("Upgrade Rule", "Placed students can continue to apply for higher categories if the CTC difference is at least Rs. 3 Lakhs. Upon receiving a second offer in a higher category, further placement processes are blocked.", "Application Rules"),
            ("Withdrawal Deadline", "Students can withdraw their application without limit only before the application deadline. No withdrawals allowed after the deadline.", "Rules for Withdrawing"),
            ("Off-Campus Offer Rule", "Students must inform the CDS Cell immediately if they receive an off-campus offer.", "Off-Campus Application Rule"),
            ("Absenteeism Debarment", "Absenteeism in any test, interview, or selection process results in debarment from the next 2 placement drives.", "Absenteeism rules and policy"),
            ("Dress Code (Normal)", "Shirt/T-shirt with collar and trousers (full pants)/jeans for boys; formal attire/trousers/jeans for girls. Flip flops are NOT permitted.", "Dress Code")
        ]
        for title, desc, category in rules:
            node_id = f"policy:{title.replace(' ', '_').lower()}"
            self._add_node(node_id, "PlacementPolicy",
                           name=title,
                           description=desc,
                           category=category,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_aipc_guidelines(self, filename: str, content: str, doc_id: str):
        node_id = "policy:aipc_guidelines"
        self._add_node(node_id, "PlacementPolicy",
                       name="AIPC Guidelines",
                       description="All India Placement Committee guidelines followed by IIT Jammu.",
                       category="AIPC Guidelines",
                       source_file=filename)
        self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_rise_up_details(self, filename: str, content: str, doc_id: str):
        node_id = "program:rise_up"
        self._add_node(node_id, "OSDEvent",
                       name="RISE-UP Summer Internship",
                       description="Research Internship in Science and Engineering for undergraduate students.",
                       eligibility="Pre-final year B.Tech/B.E./M.Sc./M.Tech students (non-IIT Jammu)",
                       stipend="Rs. 2500 per week",
                       duration="Minimum 6 weeks, up to 8 weeks",
                       source_file=filename)
        self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_cds_contact(self, filename: str, content: str, doc_id: str):
        contact_id = "contact:cds"
        self._add_node(contact_id, "SectionContact",
                       name="CDS Contact Information",
                       email="placements@iitjammu.ac.in",
                       phone="0191-2570289",
                       address="Career Development Services, IIT Jammu, Jagti, Nagrota, J&K",
                       source_file=filename)
        self._add_edge(contact_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_placement_stats(self, filename: str, content: str, doc_id: str):
        stats = [
            ("B.Tech", "Chemical Engineering", 33, 21, 10.5),
            ("B.Tech", "Civil Engineering", 25, 23, 9.5),
            ("B.Tech", "Computer Science Engineering", 35, 33, 26.6),
            ("B.Tech", "Electrical Engineering", 34, 29, 15.0),
            ("B.Tech", "Materials Engineering", 18, 17, 12.0),
            ("B.Tech", "Mechanical Engineering", 33, 26, 12.1),
            ("M.Tech", "Computer Science Engineering", 30, 30, 20.0),
            ("M.Tech", "Electrical Engineering", 34, 28, 22.8),
            ("M.Tech", "Mechanical Engineering", 12, 9, 8.8),
            ("M.Tech", "Civil Engineering", 31, 28, 6.8),
            ("M.Tech", "Chemical Engineering", 5, 4, 6.9)
        ]
        for degree, dept, registered, placed, avg in stats:
            node_id = f"placement_stat:{degree}:{dept.replace(' ', '_').lower()}:2025-26"
            self._add_node(node_id, "PlacementStat",
                           name=f"{degree} {dept} Placement 2025-26",
                           degree=degree,
                           department=dept,
                           registered=registered,
                           placed=placed,
                           avg_salary=f"{avg} LPA",
                           year="2025-26",
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")
            
        gen_stats = [
            ("2023-24", "70.8%", "53 LPA", "15.5 LPA"),
            ("2022-23", "85%", "53 LPA", "17 LPA"),
            ("2021-22", "85%", "52 LPA", "19 LPA"),
        ]
        for year, pct, highest, avg in gen_stats:
            node_id = f"placement_stat:overall:{year}"
            self._add_node(node_id, "PlacementStat",
                           name=f"Overall Placements {year}",
                           degree="Overall",
                           department="All",
                           percentage_placed=pct,
                           highest_salary=highest,
                           avg_salary=avg,
                           year=year,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    # Custom Parsers for IR
    def _parse_ir_team(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("### "):
                name = line[4:].strip()
                designation = ""
                j = i + 1
                while j < len(lines) and not lines[j].startswith("### ") and not lines[j].startswith("## "):
                    sub_line = lines[j]
                    if sub_line:
                        designation = re.sub(r'\]\(https?:.*$', '', sub_line).strip()
                        designation = designation.replace("[", "").replace("]", "").strip()
                        break
                    j += 1
                
                if name and not any(term in name.lower() for term in ("former", "dean", "head", "section", "student")):
                    resolved_name = self.resolver.resolve(name)
                    label = "SectionHead" if "Dean" in designation or "Head" in designation else "SectionPerson"
                    self._add_node(resolved_name, label,
                                   name=resolved_name,
                                   designation=designation,
                                   source_file=filename)
                    self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                    i = j - 1
            i += 1

    def _parse_ir_contact(self, filename: str, content: str, doc_id: str):
        self._parse_section_contact(filename, content, doc_id)

    def _parse_mous(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        
        countries = {
            "BRAZIL", "CANADA", "FINLAND", "FRANCE", "SOUTH KOREA", "GERMANY",
            "ITALY", "JAPAN", "TAIWAN", "NORWAY", "USA", "INTERNATIONAL"
        }
        
        current_country = "Unknown"
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.upper() in countries:
                current_country = line.upper()
                i += 1
                continue
            
            if line == "PARTNER INSTITUTION":
                if i + 2 < len(lines) and lines[i+1] == "START DATE" and lines[i+2] == "PROGRAM TYPE":
                    i += 3
                    continue
            
            if i + 2 < len(lines):
                partner = lines[i]
                start_date = lines[i+1]
                program_type = lines[i+2]
                
                if re.match(r'^\d{4}-\d{2}-\d{2}$', start_date) or start_date == "—":
                    node_id = f"mou:{partner.replace(' ', '_').lower()}"
                    self._add_node(node_id, "MOU",
                                   name=partner,
                                   partner=partner,
                                   country=current_country,
                                   start_date=start_date,
                                   program_type=program_type,
                                   source_file=filename)
                    self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")
                    i += 3
                    continue
            i += 1

    def _parse_clubs(self, filename: str, content: str, doc_id: str):
        clubs = [
            ("Coding Club", "Technical", "Focuses on competitive programming, coding competitions, and software development skills for students."),
            ("Fintech Club", "Technical", "Explores innovations in financial technology, blockchain, and digital finance solutions."),
            ("Robo-Sapiens", "Technical", "Dedicated to robotics research, automation projects, and national robotic competitions."),
            ("RE4M", "Technical", "Designing and fabricating vehicles for competitions like BAJA and Formula Student."),
            ("SAE Club", "Technical", "Society of Automotive Engineers chapter focusing on automotive design and engineering competitions."),
            ("MESH Club", "Technical & Cultural", "Bridges the gap between engineering, science, and humanities through interdisciplinary activities and projects."),
            ("Astria-Za", "Technical", "Focuses on astronomy, space exploration, and astrophysics activities and research projects."),
            ("Kritash Club", "Cultural", "Creates hope through education and social service. Working to develop awareness, help underprivileged, and overcome social challenges since 2017."),
            ("BeatStreet Club", "Cultural", "Dedicated to various dance forms, choreography, and dance performances with regular practice sessions and competitions."),
            ("Malang Club", "Cultural", "Promotes musical talents, instrument learning, vocal training, and musical performances among students."),
            ("Foot Tinkerers Club", "Cultural", "Explores culinary arts, food science, and innovative cooking techniques and food culture."),
            ("Abhivyakta Club", "Cultural", "Promotes literary activities, creative writing, poetry, storytelling, and language arts among students."),
            ("Anisoul Club", "Cultural", "Focuses on animation, digital art, graphic design, and multimedia content creation."),
            ("EBSB Club", "Cultural", "Promotes cultural unity and diversity through inter-state cultural exchange programs."),
            ("Dramatizers Club", "Cultural", "Dedicated to theatrical performances, drama productions, and acting skill development."),
            ("Wellness Club", "Cultural", "Promotes physical and mental health awareness, wellness activities, and stress management."),
            ("NAC Club", "Cultural", "Focuses on community service, social activities, and national development initiatives.")
        ]
        for name, category, desc in clubs:
            node_id = f"club:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "Club",
                           name=name,
                           category=category,
                           description=desc,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_sports(self, filename: str, content: str, doc_id: str):
        sports = [
            ("Gymnasium", "Facility", "State of the art gymnasium for student fitness."),
            ("Basketball Court", "Facility", "Outdoor basketball court with floodlights."),
            ("Cricket Ground", "Facility", "Full sized cricket ground."),
            ("Football Ground", "Facility", "Football ground with modern turf."),
            ("Inter-IIT Sports Meet", "Fest", "IIT Jammu participates in annual Inter-IIT sports meet.")
        ]
        for name, category, desc in sports:
            node_id = f"sports:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "SportsFacility" if category == "Facility" else "SportsFest",
                           name=name,
                           description=desc,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_hostels(self, filename: str, content: str, doc_id: str):
        hostels = [
            ("Canary", "Boys"),
            ("Braeg", "Boys"),
            ("Fulgar", "Boys"),
            ("Dedhar", "Girls"),
            ("Egret", "Girls")
        ]
        for name, type_gender in hostels:
            node_id = f"hostel:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "Hostel",
                           name=name,
                           gender=type_gender,
                           description=f"{name} is a {type_gender} student hostel at IIT Jammu.",
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_fests(self, filename: str, content: str, doc_id: str):
        fests = [
            ("Anhad", "Cultural & Techno Fest", "Annual flagship festival celebrating culture and technology."),
            ("Pravaah", "Sports Fest", "Annual intra-institute and inter-collegiate sports festival."),
            ("Convoquer", "Cultural Fest", "Cultural festival highlighting music, dance, and arts."),
            ("Nexus", "Technical Fest", "Technical festival showcasing robotics, coding, and engineering projects."),
            ("Pragyaan", "Academic Fest", "Science and humanities discussion fest."),
            ("Udyamitsav", "Entrepreneurship Fest", "Entrepreneurship and startup pitch fest organized by MESH.")
        ]
        for name, category, desc in fests:
            node_id = f"fest:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "Fest",
                           name=name,
                           category=category,
                           description=desc,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    # Custom Parsers for Medical Centre
    def _parse_medical_about(self, filename: str, content: str, doc_id: str):
        node_id = "medical:about"
        self._add_node(node_id, "MedicalService",
                       name="Medical Centre Info",
                       description="24x7 medical assistance with full-time doctors and emergency staff.",
                       source_file=filename)
        self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_medical_doctors(self, filename: str, content: str, doc_id: str):
        # We split by '## ' or '##' to parse each person profile block
        blocks = content.split("##")
        for block in blocks:
            if not block.strip():
                continue
            lines = [line.strip() for line in block.splitlines()]
            name_line = lines[0].strip()
            
            # Strip markdown formatting from the name if any (e.g. ### or ##)
            name_clean = re.sub(r'^[#\s\-]+', '', name_line).strip()
            if not name_clean or not name_clean.lower().startswith("dr"):
                continue
            
            designation = ""
            qualifications = ""
            experience = ""
            email = ""
            phone = ""
            
            for line in lines[1:]:
                # Extract designation/role
                if "designation" in line.lower() or "role" in line.lower():
                    match = re.search(r'(?:designation|role)\s*:\s*\**([^*:]+)', line, re.IGNORECASE)
                    if match:
                        designation = match.group(1).strip()
                # Extract qualifications
                elif "qualifications" in line.lower():
                    match = re.search(r'qualifications\s*:\s*\**([^*:]+)', line, re.IGNORECASE)
                    if match:
                        qualifications = match.group(1).strip()
                # Extract experience
                elif "experience" in line.lower():
                    match = re.search(r'experience\s*:\s*\**([^*:]+)', line, re.IGNORECASE)
                    if match:
                        experience = match.group(1).strip()
                # Extract email
                elif "email" in line.lower() or "@" in line:
                    email_match = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', line)
                    if email_match:
                        email = email_match.group(0).strip()
                # Extract phone
                elif "phone" in line.lower() or "+" in line:
                    # Strip out markdown bold / italic formatting if present
                    clean_phone_line = re.sub(r'[\*\_\:]', ' ', line)
                    phone_match = re.search(r'\+?[\d\s\-]{8,18}', clean_phone_line)
                    if phone_match:
                        phone = phone_match.group(0).strip()
            
            # Resolve name
            resolved = self.resolver.resolve(name_clean)
            
            # Check if this person is the Chairperson
            if "chairperson" in designation.lower() or "chairperson" in name_clean.lower() or "chairperson" in block.lower():
                node_id = f"head:{resolved.replace(' ', '_').lower()}"
                self._add_node(node_id, "SectionHead",
                               name=resolved,
                               designation=designation or "Chairperson - Medical Unit",
                               email=email,
                               phone=phone,
                               qualifications=qualifications,
                               experience=experience,
                               source_file=filename)
                self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")
            else:
                node_id = f"doctor:{resolved.replace(' ', '_').lower()}"
                self._add_node(node_id, "MedicalDoctor",
                               name=resolved,
                               designation=designation or "Medical Officer",
                               qualifications=qualifications,
                               experience=experience,
                               email=email,
                               phone=phone,
                               source_file=filename)
                self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")


    def _parse_medical_collaborations(self, filename: str, content: str, doc_id: str):
        hospitals = [
            ("Narayana Super Speciality Hospital", "Katra", "CGHS rates"),
            ("ASCOMS (Acharya Shri Chander College of Medical Sciences)", "Jammu", "CGHS rates"),
            ("Bee Enn General Hospital", "Jammu", "Discounted rates"),
            ("Ankur Maitrika Hospital", "Jammu", "CGHS / Discounted rates"),
            ("Fortis Hospital", "Amritsar", "Referral based CGHS rates")
        ]
        for name, location, rate_type in hospitals:
            node_id = f"hospital:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "EmpaneledHospital",
                           name=name,
                           location=location,
                           rate_type=rate_type,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_medical_contact(self, filename: str, content: str, doc_id: str):
        contact_id = "contact:medical-centre"
        self._add_node(contact_id, "SectionContact",
                       name="Medical Centre Contact Information",
                       email="medical.centre@iitjammu.ac.in",
                       phone="0191-2570636",
                       address="Health and Wellness Centre, IIT Jammu, Jagti, Nagrota, J&K",
                       source_file=filename)
        self._add_edge(contact_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_medical_services(self, filename: str, content: str, doc_id: str):
        services = {
            "dental": ("Dental Services", "Mon, Wed, Fri: 2:30 PM - 5:30 PM", "Comprehensive dental treatment."),
            "physiotherapy": ("Physiotherapy", "Monday - Friday: 9:00 AM - 5:30 PM", "Rehabilitation and physical therapy services."),
            "pharmacy": ("Pharmacy", "24x7 / Daily", "Availability of basic and prescribed medications."),
            "ambulance": ("Ambulance", "24x7", "Emergency patient transport services."),
            "ward": ("Ward Facility", "24x7", "Observation ward for student recovery."),
            "dressing": ("Dressing Room", "24x7", "First aid and dressing/minor procedure room."),
            "ecg": ("ECG Services", "Daily", "Electrocardiogram facility for cardiac monitoring."),
            "laboratory": ("Laboratory Services", "Daily", "Basic blood tests and diagnostics.")
        }
        for key, (name, timings, desc) in services.items():
            if key in filename:
                node_id = f"service:{key}"
                self._add_node(node_id, "MedicalService",
                               name=name,
                               timings=timings,
                               description=desc,
                               source_file=filename)
                self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    # Custom Parsers for OSD
    def _parse_osd_team(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("### ") and not any(term in line.lower() for term in ("leadership", "associate dean", "head of section", "staff")):
                designation = line[4:].strip()
                name = ""
                j = i + 1
                while j < len(lines) and not lines[j].startswith("### ") and not lines[j].startswith("## "):
                    sub_line = lines[j]
                    if sub_line.startswith("#### "):
                        name = sub_line[5:].strip()
                        break
                    j += 1
                if name:
                    resolved_name = self.resolver.resolve(name)
                    label = "SectionHead" if "Dean" in designation or "HoS" in designation or "Head" in designation else "SectionPerson"
                    self._add_node(resolved_name, label,
                                   name=resolved_name,
                                   designation=designation,
                                   source_file=filename)
                    self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                    i = j - 1
            i += 1

    def _parse_osd_uba(self, filename: str, content: str, doc_id: str):
        node_id = "uba:program"
        self._add_node(node_id, "UBAProgram",
                       name="Unnat Bharat Abhiyan",
                       description="UBA is a flagship program of MoE, aiming to link higher education institutions with rural communities.",
                       focus_areas="Water management, organic farming, sanitation, digital literacy",
                       coordinator="Dr. Sameer Kumar Sarma Pachalla",
                       source_file=filename)
        self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_osd_ces(self, filename: str, content: str, doc_id: str):
        node_id = "ces:program"
        self._add_node(node_id, "CESProgram",
                       name="Centre for Essential Skills",
                       description="CES aims to provide technical and vocational skilling programs for youth and community.",
                       courses="Web development, CNC programming, welding, office productivity",
                       source_file=filename)
        self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_osd_events(self, filename: str, content: str, doc_id: str):
        events = [
            ("RAISE", "OSD Outreach Program", "Skills training and school interaction workshops."),
            ("RISE-UP", "OSD Skilling Initiative", "Undergraduate research internship program co-managed with CDS.")
        ]
        for name, category, desc in events:
            node_id = f"osd_event:{name.replace(' ', '_').lower()}"
            self._add_node(node_id, "OSDEvent",
                           name=name,
                           category=category,
                           description=desc,
                           source_file=filename)
            self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_osd_contact(self, filename: str, content: str, doc_id: str):
        contact_id = "contact:osd"
        self._add_node(contact_id, "SectionContact",
                       name="OSD Contact Information",
                       email="osd@iitjammu.ac.in",
                       phone="0191-2570636",
                       address="Outreach and Skilling Division, IIT Jammu, Jagti, Nagrota, J&K",
                       source_file=filename)
        self._add_edge(contact_id, doc_id, "SOURCE_DOCUMENT")

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
        elif label == "AlumniMedalist":
            parts.append(f"{name} is an IIT Jammu Alumni medalist.")
            if data.get("award"):
                parts.append(f"Award/Medal: {data['award']}.")
            if data.get("convocation"):
                parts.append(f"Convocation: {data['convocation']}.")
            if data.get("year"):
                parts.append(f"Graduation Year: {data['year']}.")
            if data.get("degree"):
                parts.append(f"Degree: {data['degree']}.")
            if data.get("department"):
                parts.append(f"Department: {data['department']}.")
        elif label == "AlumniAward":
            parts.append(f"{name} received the Alumni Award at IIT Jammu.")
            if data.get("award_name"):
                parts.append(f"Award: {data['award_name']}.")
            if data.get("year"):
                parts.append(f"Year: {data['year']}.")
            if data.get("degree"):
                parts.append(f"Degree: {data['degree']}.")
            if data.get("department"):
                parts.append(f"Department: {data['department']}.")
        elif label == "Recruiter":
            parts.append(f"{name} is a past recruiter/company that visited IIT Jammu for campus placements.")
        elif label == "PlacementPolicy":
            parts.append(f"Placement Policy - {name}.")
            if data.get("description"):
                parts.append(f"Policy details: {data['description']}.")
            if data.get("category"):
                parts.append(f"Category: {data['category']}.")
        elif label == "PlacementStat":
            parts.append(f"Placement Statistics for {name}.")
            if data.get("degree"):
                parts.append(f"Degree: {data['degree']}.")
            if data.get("department"):
                parts.append(f"Department: {data['department']}.")
            if data.get("registered"):
                parts.append(f"Registered Students: {data['registered']}.")
            if data.get("placed"):
                parts.append(f"Placed Students: {data['placed']}.")
            if data.get("percentage_placed"):
                parts.append(f"Placement Percentage: {data['percentage_placed']}.")
            if data.get("highest_salary"):
                parts.append(f"Highest CTC: {data['highest_salary']}.")
            if data.get("avg_salary"):
                parts.append(f"Average CTC: {data['avg_salary']}.")
            if data.get("year"):
                parts.append(f"Academic/Batch Year: {data['year']}.")
        elif label == "MOU":
            parts.append(f"MOU partnership between IIT Jammu and {name}.")
            if data.get("partner"):
                parts.append(f"Partner Institution: {data['partner']}.")
            if data.get("country"):
                parts.append(f"Country: {data['country']}.")
            if data.get("program_type"):
                parts.append(f"Collaboration Type: {data['program_type']}.")
        elif label == "Club":
            parts.append(f"{name} is a student club at IIT Jammu.")
            if data.get("category"):
                parts.append(f"Category: {data['category']} Club.")
            if data.get("description"):
                parts.append(f"Description: {data['description']}.")
        elif label in ("SportsFacility", "SportsFest"):
            parts.append(f"{name} is a sports facility or event at IIT Jammu.")
            if data.get("description"):
                parts.append(f"Details: {data['description']}.")
        elif label == "Hostel":
            parts.append(f"{name} Hostel is a residential facility at IIT Jammu.")
            if data.get("gender"):
                parts.append(f"Type: {data['gender']}' hostel.")
            if data.get("description"):
                parts.append(f"Details: {data['description']}.")
        elif label == "Fest":
            parts.append(f"{name} is an annual festival/event at IIT Jammu.")
            if data.get("category"):
                parts.append(f"Category: {data['category']}.")
            if data.get("description"):
                parts.append(f"Description: {data['description']}.")
        elif label == "MedicalDoctor":
            parts.append(f"{name} is a doctor/medical officer at the IIT Jammu Medical Centre.")
            if data.get("designation"):
                parts.append(f"Designation: {data['designation']}.")
            if data.get("qualifications"):
                parts.append(f"Qualifications: {data['qualifications']}.")
            if data.get("experience"):
                parts.append(f"Experience: {data['experience']}.")
            if data.get("email"):
                parts.append(f"Email: {data['email']}.")
            if data.get("phone"):
                parts.append(f"Phone: {data['phone']}.")
        elif label == "EmpaneledHospital":
            parts.append(f"{name} is an empaneled hospital of IIT Jammu.")
            if data.get("location"):
                parts.append(f"Location: {data['location']}.")
            if data.get("rate_type"):
                parts.append(f"Rates/Coverage: {data['rate_type']}.")
        elif label == "MedicalService":
            parts.append(f"Medical service - {name} at the IIT Jammu Health & Wellness Centre.")
            if data.get("timings"):
                parts.append(f"Timings: {data['timings']}.")
            if data.get("description"):
                parts.append(f"Description: {data['description']}.")
        elif label == "UBAProgram":
            parts.append(f"{name} program at IIT Jammu.")
            if data.get("description"):
                parts.append(f"Details: {data['description']}.")
            if data.get("focus_areas"):
                parts.append(f"Focus Areas: {data['focus_areas']}.")
            if data.get("coordinator"):
                parts.append(f"Coordinator: {data['coordinator']}.")
        elif label == "CESProgram":
            parts.append(f"CES Program: {name} at IIT Jammu.")
            if data.get("description"):
                parts.append(f"Details: {data['description']}.")
            if data.get("courses"):
                parts.append(f"Courses: {data['courses']}.")
        elif label == "OSDEvent":
            parts.append(f"OSD Event: {name} at IIT Jammu.")
            if data.get("category"):
                parts.append(f"Category: {data['category']}.")
            if data.get("description"):
                parts.append(f"Details: {data['description']}.")
        elif label == "AcademicProgram":
            parts.append(f"{name} is an Academic Program offered by the Academics section at IIT Jammu.")
            if data.get("level"):
                parts.append(f"Program Level: {data['level']}.")
            if data.get("superseded"):
                parts.append("Note: This program version/curriculum is superseded by a newer version.")
        elif label == "Specialization":
            parts.append(f"{name} is an academic Specialization/Minor/Honours program offered by the Academics section at IIT Jammu.")
            if data.get("type"):
                parts.append(f"Specialization Type: {data['type']}.")
            if data.get("level"):
                parts.append(f"Program Level: {data['level']}.")
            if data.get("superseded"):
                parts.append("Note: This specialization version/curriculum is superseded by a newer version.")
        elif label == "Course":
            parts.append(f"Course: {name}.")
            if data.get("code"):
                parts.append(f"Course Code: {data['code']}.")
            if data.get("ltp"):
                parts.append(f"L-T-P Structure: {data['ltp']}.")
            if data.get("credits"):
                parts.append(f"Credits: {data['credits']}.")
        elif label == "ElectiveBucket":
            parts.append(f"Elective Bucket/Track: {name}.")
        elif label == "CommitteeMember":
            parts.append(f"{name} is a member of the {data.get('committee_type', 'committee')} ({data.get('committee_name', '')}) of {data.get('department', '')} department at IIT Jammu.")
            if data.get("designation"):
                parts.append(f"Role/Designation: {data['designation']}.")
            if data.get("notification_date"):
                parts.append(f"Notification Date: {data['notification_date']}.")
        elif label == "FacultyAdvisor":
            parts.append(f"{name} is the Faculty Advisor for {data.get('programme', '')} (Batch Year: {data.get('batch_year', '')}) at IIT Jammu.")
            if data.get("notification_date"):
                parts.append(f"Notification Date: {data['notification_date']}.")
        elif label == "ProgramCoordinator":
            parts.append(f"{name} is the Programme Coordinator for {data.get('programme', '')} (Batch Year: {data.get('batch_year', '')}) at IIT Jammu.")
            if data.get("notification_date"):
                parts.append(f"Notification Date: {data['notification_date']}.")
        elif label == "FeeStructure":
            parts.append(f"Fee structure for {data.get('programme', '')} (Entry/Admission Year: {data.get('entry_year', '')}).")
            if data.get("income_category"):
                parts.append(f"Income Category: {data['income_category']}.")
            if data.get("fee_gen_obc_ews"):
                parts.append(f"Tuition Fee & Other Charges (General/OBC/EWS): {data['fee_gen_obc_ews']}.")
            if data.get("fee_sc_st_pwd"):
                parts.append(f"SC/ST/PwD Charges: {data['fee_sc_st_pwd']}.")
        else:
            parts.append(f"{name} ({label}) in {section_name} section.")

        if parts:
            descriptions.append({
                "id": node_id,
                "text": " ".join(parts),
                "metadata": {"label": label, "name": name}
            })
    return descriptions

