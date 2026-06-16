from graphrag.rules_parser import RulesParser
import os
import re

filepath = "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/UG_Curriculum_2022_Scheme_IIT_Jammu.md"
program = "undergraduate_2022"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split("\n")
# Skip YAML header
yaml_lines = 0
if lines and lines[0].strip() == "---":
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_lines = i + 1
            break

content_lines = lines[yaml_lines:]

parser = RulesParser()

has_toc = any("Contents" in line or "CONTENTS" in line for line in content_lines[:400])
body_started = not has_toc

active_path = []
for idx, line in enumerate(content_lines):
    line_str = line.strip()
    if not line_str:
        continue
    
    # Detect start of body
    if not body_started:
        if re.match(r"^\s*1\s+[A-Za-z]", line_str):
            if not re.search(r"\d+$", line_str) and "..." not in line_str and " . ." not in line_str:
                body_started = True
                print(f"Body started at line {idx + yaml_lines + 1}: {line_str}")
        if not body_started:
            continue

    sec_num_str = None
    title_str = None
    
    m_num = parser.num_pat.match(line_str)
    m_let = parser.let_pat.match(line_str)
    m_md = parser.md_pat.match(line_str)
    
    if m_num and not parser.is_toc_line(line_str):
        sec_num_str, title_str = m_num.groups()
    elif m_let and not parser.is_toc_line(line_str):
        sec_num_str, title_str = m_let.groups()
    elif m_md and not parser.is_toc_line(line_str):
        sec_num_str, title_str = m_md.groups()
        if not sec_num_str:
            sec_num_str = ""

    is_heading = False
    sec_num = None
    if sec_num_str and title_str:
        title_clean = title_str.strip().rstrip("]").rstrip(")").strip()
        is_abbreviation = any(title_clean.endswith(abbr) for abbr in ("B.Tech.", "M.Tech.", "Ph.D.", "B.T.", "M.T.", "Dr.", "Prof.", "i.e.", "e.g.", "etc."))
        if (not (title_clean.endswith(".") and not is_abbreviation) and 
            not title_clean.endswith(";") and 
            not title_clean.endswith(",") and 
            "|" not in line_str and
            len(title_clean) < 100 and 
            len(title_clean) >= 2):
            
            sec_num = parser.parse_sec_num(sec_num_str)
            if sec_num:
                if len(sec_num) == 1 or (len(sec_num) == 2 and sec_num[0] == 18):
                    if program == "phd" and len(sec_num) == 1:
                        if not sec_num_str.rstrip(".")[0].isalpha():
                            continue
                    
                    use_mixed_case = program in ("undergraduate_2022", "phd")
                    if use_mixed_case:
                        if len(title_clean.split()) > 6 or not title_clean[0].isupper():
                            continue
                    else:
                        letters = [c for c in title_clean if c.isalpha()]
                        if letters:
                            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                            if upper_ratio < 0.7:
                                continue
                
                if sec_num[0] == 18 and not any(p[0][0] == 18 for p in active_path if p[0]):
                    active_path = [((18,), "root")]
                
                if not active_path:
                    if sec_num == (1,):
                        is_heading = True
                else:
                    current_num, _ = active_path[-1]
                    if len(sec_num) == len(current_num) + 1 and sec_num[:-1] == current_num:
                        is_heading = True
                    elif len(sec_num) <= len(current_num) and len(sec_num) - 1 < len(active_path):
                        ancestor, _ = active_path[len(sec_num)-1]
                        if len(sec_num) == 1:
                            if sec_num[0] == active_path[0][0][0] + 1:
                                is_heading = True
                        elif len(sec_num) == 2 and sec_num[0] == 18:
                            if len(ancestor) >= 2:
                                if sec_num[1] > ancestor[1]:
                                    is_heading = True
                            else:
                                is_heading = True
                        else:
                            if sec_num[-1] > ancestor[-1] and sec_num[:-1] == ancestor[:-1]:
                                is_heading = True
                                
                if sec_num_str in ("2.2.2", "2.2.2.1") or (idx + yaml_lines + 1) in (4056, 4080):
                    print(f"DEBUG line {idx + yaml_lines + 1}: sec_num_str='{sec_num_str}', title='{title_clean}'")
                    print(f"  active_path: {active_path}")
                    print(f"  is_heading: {is_heading}")
                                
    if is_heading and sec_num:
        sec_id = f"sec_{sec_num_str}"
        active_path = [p for p in active_path if len(p[0]) < len(sec_num)]
        active_path.append((sec_num, sec_id))
        print(f"ACCEPTED heading {sec_num_str} -> {title_clean} (active_path: {[p[0] for p in active_path]})")
