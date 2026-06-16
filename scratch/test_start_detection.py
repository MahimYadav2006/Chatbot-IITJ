import re

files = [
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md", "undergraduate"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/UG_Curriculum_2022_Scheme_IIT_Jammu.md", "undergraduate_2022"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/MTech/M.Tech_RRs___Curric..md", "mtech"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/PhD/PhD_RRs.md", "phd")
]

for filepath, prog in files:
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Check if has toc (look up to 400 lines)
    has_toc = any("Contents" in line or "CONTENTS" in line for line in lines[:400])
    
    # Find start line
    start_line_idx = -1
    if has_toc:
        for idx, line in enumerate(lines):
            line_str = line.strip()
            # Must start with 1, followed by space and words
            if re.match(r"^\s*1\s+[A-Za-z]", line_str):
                # Must not end with digit or contain dots
                if not re.search(r"\d+$", line_str) and "..." not in line_str and " . ." not in line_str:
                    start_line_idx = idx
                    break
    else:
        # For PhD or other files, start at line 12 or so
        start_line_idx = 10
        
    print(f"File: {filepath.split('/')[-1]}")
    print(f"  has_toc: {has_toc}")
    print(f"  Starts at line {start_line_idx + 1}: {repr(lines[start_line_idx].strip()) if start_line_idx != -1 else 'NOT FOUND'}")
