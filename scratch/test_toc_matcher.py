import re
import os

def is_valid_sec_num(s):
    s = s.strip().rstrip(".")
    if not s:
        return False
    parts = s.split(".")
    for p in parts:
        if not (p.isdigit() or (len(p) == 1 and p.isalpha())):
            return False
    return True

def extract_toc_headings(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    has_toc = any("Contents" in line or "CONTENTS" in line for line in lines[:400])
    if not has_toc:
        return None
        
    toc_headings = {}
    
    body_start_idx = -1
    for idx, line in enumerate(lines):
        line_str = line.strip()
        if re.match(r"^\s*1\s+[A-Za-z]", line_str):
            if not re.search(r"\d+$", line_str) and "..." not in line_str and " . ." not in line_str:
                body_start_idx = idx
                break
                
    if body_start_idx == -1:
        return None
        
    toc_lines = lines[:body_start_idx]
    toc_pat = re.compile(r"^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+(.*?)(?:\s*[\.\s\-\)]+\s*\d+|\s+ee\s+\d+|\s*\]\s*[\.\s]*\d+)?$")
    
    for line in toc_lines:
        line_str = line.strip()
        if not line_str or "Contents" in line_str or "CONTENTS" in line_str:
            continue
        if line_str == "---" or line_str.startswith("source_") or line_str.startswith("category:") or line_str.startswith("converted_") or line_str.startswith("subcategory:") or line_str.startswith("institution:") or line_str.startswith("document_type:"):
            continue
        if line_str.startswith("<!-- Page") or line_str.startswith("---") or line_str.startswith("\f"):
            continue
            
        m = toc_pat.match(line_str)
        if m:
            sec_num_str, title_str = m.groups()
            sec_num_str = sec_num_str.strip().rstrip(".")
            
            if not is_valid_sec_num(sec_num_str):
                continue
                
            title_str = title_str.strip()
            title_str = re.sub(r"[\s\.\-\)\]]+$", "", title_str)
            title_str = title_str.strip()
            
            if len(title_str) < 2 or len(title_str) > 100:
                continue
                
            toc_headings[sec_num_str] = title_str
            
    return toc_headings, body_start_idx

def titles_match(toc_title, body_title):
    def norm(t):
        return re.sub(r"[^a-z0-9]", "", t.lower())
    t1 = norm(toc_title)
    t2 = norm(body_title)
    if not t1 or not t2:
        return False
    if t1 in t2 or t2 in t1:
        return True
    w1 = set(re.findall(r"\w+", toc_title.lower()))
    w2 = set(re.findall(r"\w+", body_title.lower()))
    if not w1 or not w2:
        return False
    overlap = w1.intersection(w2)
    min_len = min(len(w1), len(w2))
    if len(overlap) / min_len >= 0.4:
        return True
    return False

filepath = "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md"

toc_info = extract_toc_headings(filepath)
if toc_info:
    toc_headings, body_start_idx = toc_info
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    content_lines = lines[body_start_idx:]
    heading_pat = re.compile(r"^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+([A-Za-z0-9_ &,\-\(\)\/\.\'\’\”\[\]]+)$")
    
    detected = []
    for idx, line in enumerate(content_lines):
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("<!-- Page") or line_str.startswith("---") or line_str.startswith("\f"):
            continue
            
        m = heading_pat.match(line_str)
        if m:
            sec_num_str, title_str = m.groups()
            sec_num_str = sec_num_str.strip().rstrip(".")
            
            if not is_valid_sec_num(sec_num_str):
                continue
                
            title_str = title_str.strip().rstrip("]").rstrip(")").strip()
            
            if sec_num_str in toc_headings:
                toc_title = toc_headings[sec_num_str]
                if titles_match(toc_title, title_str):
                    detected.append((idx + body_start_idx + 1, sec_num_str, title_str))
                    
    print(f"Detected {len(detected)} matched headings:")
    for line_num, sec_num, title in detected:
        print(f"Line {line_num}: {sec_num} -> {title}")
else:
    print("Failed to parse TOC info.")
