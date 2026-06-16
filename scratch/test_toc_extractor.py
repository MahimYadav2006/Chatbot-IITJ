import re

def extract_toc_headings(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Find TOC lines
    # TOC exists between first Page and the start of the body
    has_toc = any("Contents" in line or "CONTENTS" in line for line in lines[:400])
    if not has_toc:
        return None
        
    toc_headings = {}
    
    # We stop collecting when the body starts
    body_start_idx = -1
    for idx, line in enumerate(lines):
        line_str = line.strip()
        if re.match(r"^\s*1\s+[A-Za-z]", line_str):
            if not re.search(r"\d+$", line_str) and "..." not in line_str and " . ." not in line_str:
                body_start_idx = idx
                break
                
    if body_start_idx == -1:
        return None
        
    # Parse TOC lines
    toc_lines = lines[:body_start_idx]
    # Regex to match TOC lines like:
    # 1.1 Senate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
    # 1.2.1 Purview of the SUGB . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
    # 1.3.1 Purview of the DUGC) 3
    toc_pat = re.compile(r"^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+(.*?)(?:\s*[\.\s\-\)]+\s*\d+|\s+ee\s+\d+|\s*\]\s*[\.\s]*\d+)?$")
    
    for line in toc_lines:
        line_str = line.strip()
        if not line_str or "Contents" in line_str or "CONTENTS" in line_str:
            continue
        # Skip YAML header
        if line_str == "---" or line_str.startswith("source_") or line_str.startswith("category:") or line_str.startswith("converted_") or line_str.startswith("subcategory:") or line_str.startswith("institution:") or line_str.startswith("document_type:"):
            continue
        if line_str.startswith("<!-- Page") or line_str.startswith("---") or line_str.startswith("\f"):
            continue
            
        m = toc_pat.match(line_str)
        if m:
            sec_num_str, title_str = m.groups()
            sec_num_str = sec_num_str.strip().rstrip(".")
            # Clean title
            title_str = title_str.strip()
            # Strip trailing dots or page numbers if any remain
            title_str = re.sub(r"[\s\.\-\)\]]+$", "", title_str)
            title_str = title_str.strip()
            
            # Filter out obvious non-headings
            if len(title_str) < 2 or len(title_str) > 100:
                continue
                
            toc_headings[sec_num_str] = title_str
            
    return toc_headings

# Test on our files
files = [
    "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md",
    "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/UG_Curriculum_2022_Scheme_IIT_Jammu.md",
    "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/MTech/M.Tech_RRs___Curric..md"
]

for fp in files:
    headings = extract_toc_headings(fp)
    print("=" * 60)
    print(f"File: {fp.split('/')[-1]}")
    if headings:
        print(f"Extracted {len(headings)} TOC headings:")
        for num, title in sorted(headings.items()):
            print(f"  {num} -> {title}")
    else:
        print("No TOC found or failed to parse.")
