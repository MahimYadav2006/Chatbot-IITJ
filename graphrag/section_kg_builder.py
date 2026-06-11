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
        else:
            parts.append(f"{name} ({label}) in {section_name} section.")

        if parts:
            descriptions.append({
                "id": node_id,
                "text": " ".join(parts),
                "metadata": {"label": label, "name": name}
            })
    return descriptions

