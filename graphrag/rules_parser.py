import re
import os
import logging
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class RulesParser:
    def __init__(self):
        # Heading regex patterns
        # 1. Numeric headings: "1.1 Senate" or "1.2.2. Chairperson"
        self.num_pat = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*\.?)\s+([A-Za-z0-9_ &,\-\(\)\/\.\'\’\”\[\]]+)$")
        # 2. Letter headings: "A. Senate" or "B.1 Purview"
        self.let_pat = re.compile(r"^\s*([A-Z]+(?:\.[0-9]+)*\.?)\s+([A-Za-z0-9_ &,\-\(\)\/\.\'\’\”\[\]]+)$")
        # 3. Markdown headings: "## 1.1 Senate"
        self.md_pat = re.compile(r"^\s*#+\s+([A-Za-z0-9\.]+)?\s*(.*)$")

    def parse_sec_num(self, s: str) -> Optional[Tuple[int, ...]]:
        """Convert section number string to tuple of ints for hierarchy comparisons."""
        s = s.strip().rstrip(".")
        if s.upper().startswith("R."):
            parts = [18]
            for p in s[2:].split("."):
                if not p:
                    continue
                try:
                    parts.append(int(p))
                except ValueError:
                    return None
            return tuple(parts)
            
        parts = []
        for p in s.split("."):
            try:
                parts.append(int(p))
            except ValueError:
                # Handle letters (e.g. A -> 1, B -> 2)
                if len(p) == 1 and p.isalpha():
                    parts.append(ord(p.upper()) - ord("A") + 1)
                else:
                    return None
        return tuple(parts)

    def is_toc_line(self, line: str) -> bool:
        """Check if a line looks like it belongs to Table of Contents."""
        line = line.strip()
        if not line:
            return False
        # Common TOC patterns in parsed PDFs
        if "..." in line or ".. " in line or " . ." in line:
            return True
        if line.endswith(" ee"):
            return True
        if "]" in line and any(c.isdigit() for c in line[-5:]):
            return True
        return False

    def clean_text(self, text: str) -> str:
        """Clean running headers and footer text."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line_str = line.strip()
            # Skip page markers
            if line_str.startswith("<!-- Page") or line_str.startswith("---") or line_str.startswith("\f"):
                continue
            # Skip running headers/footers
            if "ACADEMIC ADMINISTRATION" in line_str and ("Page" in line_str or "of" in line_str):
                continue
            if "Academic Rules, Curriculum" in line_str or "IIT-Jammu: Academic Rules" in line_str:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _recover_missing_top_level_sections(
        self,
        content_lines: List[str],
        sections: List[Dict[str, Any]],
        program: str,
        source_file: str,
    ) -> List[Dict[str, Any]]:
        """Recover mixed-case top-level sections skipped after PDF page headers."""
        existing_top = {
            int(sec["section_number"].strip().rstrip("."))
            for sec in sections
            if sec.get("section_number", "").strip().rstrip(".").isdigit()
        }
        existing_top_positions = {}
        for sec in sections:
            sec_number = sec.get("section_number", "").strip().rstrip(".")
            if not sec_number.isdigit():
                continue
            title = sec.get("title", "").strip()
            for idx, line in enumerate(content_lines):
                if re.match(rf"^\s*{re.escape(sec_number)}\.?\s+{re.escape(title)}\s*$", line.strip()):
                    existing_top_positions[int(sec_number)] = idx
                    break

        candidates = []
        heading_pat = re.compile(r"^\s*([0-9]+)\s+([A-Z][A-Za-z0-9_ &,\-\(\)\/\.\'\’]+)$")

        for idx, line in enumerate(content_lines):
            line_str = line.strip()
            match = heading_pat.match(line_str)
            if not match or self.is_toc_line(line_str):
                continue
            sec_number, title = match.groups()
            number = int(sec_number)
            title = title.strip()
            if number in existing_top or title.isupper() or len(title.split()) > 6:
                continue
            previous_numbers = [n for n in existing_top_positions if n < number]
            if previous_numbers:
                previous_number = max(previous_numbers)
                if idx <= existing_top_positions[previous_number]:
                    continue
            candidates.append((idx, sec_number, title))

        deduped_candidates = {}
        for candidate in candidates:
            deduped_candidates[candidate[1]] = candidate
        candidates = sorted(deduped_candidates.values(), key=lambda item: item[0])

        for cand_idx, (line_idx, sec_number, title) in enumerate(candidates):
            next_idx = len(content_lines)
            for later_idx, later_sec_number, _ in candidates[cand_idx + 1:]:
                if int(later_sec_number) > int(sec_number):
                    next_idx = later_idx
                    break

            body_lines = content_lines[line_idx + 1:next_idx]
            full_text = self.clean_text("\n".join(body_lines))
            if not full_text:
                continue

            section_id = f"{program}_sec_{sec_number}_recovered"
            sections.append({
                "id": section_id,
                "section_number": sec_number,
                "title": title,
                "full_text": full_text,
                "parent_id": f"{program}_root",
                "program": program,
                "source_file": source_file,
            })

        return sections

    def _recover_mtech_appendix_sections(
        self,
        content_lines: List[str],
        sections: List[Dict[str, Any]],
        program: str,
        source_file: str,
    ) -> List[Dict[str, Any]]:
        """Recover M.Tech appendix sections that OCR/parser rules swallow into section 13."""
        if program != "mtech":
            return sections

        appendix_numbers = {"A.1", "A.2", "A.3", "B"}
        sections = [
            sec for sec in sections
            if sec.get("section_number", "").strip().rstrip(".").upper() not in appendix_numbers
        ]
        existing = {
            sec.get("section_number", "").strip().rstrip(".").upper()
            for sec in sections
        }
        appendix_headings = []
        heading_patterns = (
            re.compile(r"^\s*(A\.[123])[\._]?\s+(.+?)\s*$", re.IGNORECASE),
            re.compile(r"^\s*B_+\s*Course code convention[,]?\s*$", re.IGNORECASE),
        )

        for idx, line in enumerate(content_lines):
            line_str = line.strip()
            if not line_str or self.is_toc_line(line_str):
                continue

            match = heading_patterns[0].match(line_str)
            if match:
                sec_number = match.group(1).upper()
                title = match.group(2).strip().rstrip(",")
                appendix_headings.append((idx, sec_number, title))
                continue

            if heading_patterns[1].match(line_str):
                appendix_headings.append((idx, "B", "Course code convention"))

        deduped = {}
        for item in appendix_headings:
            deduped[item[1]] = item
        appendix_headings = sorted(deduped.values(), key=lambda item: item[0])

        for heading_idx, (line_idx, sec_number, title) in enumerate(appendix_headings):
            if sec_number in existing:
                continue

            next_idx = (
                appendix_headings[heading_idx + 1][0]
                if heading_idx + 1 < len(appendix_headings)
                else len(content_lines)
            )
            full_text = self.clean_text("\n".join(content_lines[line_idx + 1:next_idx]))
            if not full_text:
                continue

            section_id = f"{program}_appendix_{sec_number.lower().replace('.', '_')}"
            sections.append({
                "id": section_id,
                "section_number": sec_number,
                "title": title,
                "full_text": full_text,
                "parent_id": f"{program}_root",
                "program": program,
                "source_file": source_file,
            })

        return sections

    def parse_file(self, filepath: str, program: str) -> List[Dict[str, Any]]:
        """Parse a rules markdown file into hierarchical sections."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into lines and skip YAML header
        lines = content.split("\n")
        yaml_lines = 0
        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    yaml_lines = i + 1
                    break
        
        content_lines = lines[yaml_lines:]
        
        # Determine if file has a TOC to skip it
        has_toc = any("Contents" in line or "CONTENTS" in line for line in content_lines[:400])
        body_started = not has_toc
        
        sections = []
        current_section = {
            "id": f"{program}_root",
            "section_number": "0",
            "title": "Root",
            "full_text": "",
            "parent_id": None,
            "program": program,
            "source_file": os.path.basename(filepath)
        }
        
        active_path = [] # list of (sec_num_tuple, section_id)
        
        section_counter = 0
        
        for line in content_lines:
            line_str = line.strip()
            if not line_str:
                if current_section["full_text"]:
                    current_section["full_text"] += "\n"
                continue
            
            # Skip page markers
            if line_str.startswith("<!-- Page") or line_str.startswith("---") or line_str.startswith("\f"):
                continue

            # Detect start of body
            if not body_started:
                if re.match(r"^\s*1\s+[A-Za-z]", line_str):
                    if not re.search(r"\d+$", line_str) and "..." not in line_str and " . ." not in line_str:
                        body_started = True
                if not body_started:
                    continue

            # Check if line matches a heading pattern
            sec_num_str = None
            title_str = None
            
            # Try numeric pattern first
            m_num = self.num_pat.match(line_str)
            m_let = self.let_pat.match(line_str)
            m_md = self.md_pat.match(line_str)
            
            if m_num and not self.is_toc_line(line_str):
                sec_num_str, title_str = m_num.groups()
            elif m_let and not self.is_toc_line(line_str):
                sec_num_str, title_str = m_let.groups()
            elif m_md and not self.is_toc_line(line_str):
                # If markdown hash format
                sec_num_str, title_str = m_md.groups()
                if not sec_num_str:
                    sec_num_str = ""
            
            is_heading = False
            sec_num = None
            
            if sec_num_str and title_str:
                title_str = title_str.strip().rstrip("]").rstrip(")").strip()
                is_abbreviation = any(title_str.endswith(abbr) for abbr in ("B.Tech.", "M.Tech.", "Ph.D.", "B.T.", "M.T.", "Dr.", "Prof.", "i.e.", "e.g.", "etc."))
                if (not (title_str.endswith(".") and not is_abbreviation) and 
                    not title_str.endswith(";") and 
                    not title_str.endswith(",") and 
                    "|" not in line_str and
                    len(title_str) < 100 and 
                    len(title_str) >= 2):
                    
                    sec_num = self.parse_sec_num(sec_num_str)
                    if sec_num:
                        # LEVEL 1 HEADING RULES
                        if len(sec_num) == 1 or (len(sec_num) == 2 and sec_num[0] == 18):
                            if program == "phd" and len(sec_num) == 1:
                                if not sec_num_str.rstrip(".")[0].isalpha():
                                    continue
                            
                            use_mixed_case = program in ("undergraduate_2022", "phd")
                            if use_mixed_case:
                                if len(title_str.split()) > 6 or not title_str[0].isupper():
                                    continue
                            else:
                                letters = [c for c in title_str if c.isalpha()]
                                if letters:
                                    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                                    is_sequential_top = (
                                        len(sec_num) == 1
                                        and title_str[0].isupper()
                                        and len(title_str.split()) <= 6
                                        and (
                                            not active_path
                                            or sec_num[0] == active_path[0][0][0] + 1
                                        )
                                    )
                                    if upper_ratio < 0.7 and not is_sequential_top:
                                        continue
                        
                        # Auto-initialize regulations namespace
                        if sec_num[0] == 18 and not any(p[0][0] == 18 for p in active_path if p[0]):
                            active_path = [((18,), f"{program}_root")]
                        
                        # Validate transition to avoid list item confusion
                        if not active_path:
                            # Start path
                            if sec_num == (1,):
                                is_heading = True
                        else:
                            current_num, _ = active_path[-1]
                            # 1. Depth increase
                            if len(sec_num) == len(current_num) + 1 and sec_num[:-1] == current_num:
                                is_heading = True
                            # 2. Sibling or ancestor sibling transition
                            elif len(sec_num) <= len(current_num) and len(sec_num) - 1 < len(active_path):
                                ancestor, _ = active_path[len(sec_num)-1]
                                if len(sec_num) == 1:
                                    if sec_num[0] == active_path[0][0][0] + 1:
                                        is_heading = True
                                elif len(sec_num) == 2 and sec_num[0] == 18:
                                    # Regulations sibling transition
                                    if len(ancestor) >= 2:
                                        if sec_num[1] > ancestor[1]:
                                            is_heading = True
                                    else:
                                        is_heading = True
                                else:
                                    if sec_num[-1] > ancestor[-1] and sec_num[:-1] == ancestor[:-1]:
                                        is_heading = True
            
            if is_heading and sec_num:
                # Save previous section
                if current_section["full_text"]:
                    current_section["full_text"] = self.clean_text(current_section["full_text"])
                    sections.append(current_section)
                
                # Determine parent_id
                parent_id = None
                # Truncate active path to appropriate parent length
                active_path = [p for p in active_path if len(p[0]) < len(sec_num)]
                if active_path:
                    _, parent_id = active_path[-1]
                else:
                    parent_id = f"{program}_root"
                
                # Create new section
                section_counter += 1
                sec_id = f"{program}_sec_{sec_num_str.replace('.', '_')}_{section_counter}"
                
                current_section = {
                    "id": sec_id,
                    "section_number": sec_num_str,
                    "title": title_str,
                    "full_text": "",
                    "parent_id": parent_id,
                    "program": program,
                    "source_file": os.path.basename(filepath)
                }
                
                # Update active path
                active_path.append((sec_num, sec_id))
            else:
                # Append line to text of current section
                if current_section["full_text"]:
                    current_section["full_text"] += "\n"
                current_section["full_text"] += line
        
        # Save last section
        if current_section["full_text"]:
            current_section["full_text"] = self.clean_text(current_section["full_text"])
            sections.append(current_section)

        sections = self._recover_missing_top_level_sections(
            content_lines=content_lines,
            sections=sections,
            program=program,
            source_file=os.path.basename(filepath),
        )
        sections = self._recover_mtech_appendix_sections(
            content_lines=content_lines,
            sections=sections,
            program=program,
            source_file=os.path.basename(filepath),
        )

        return sections
