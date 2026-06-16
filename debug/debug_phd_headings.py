import re

filepath = "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/PhD/PhD_RRs.md"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

yaml_lines = 0
if lines[0].strip() == "---":
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_lines = i + 1
            break

content_lines = lines[yaml_lines:]

def parse_sec_num(s):
    s = s.strip().rstrip(".")
    parts = []
    for p in s.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            if len(p) == 1 and p.isalpha():
                parts.append(ord(p.upper()) - ord("A") + 1)
            else:
                return None
    return tuple(parts)

heading_pat = re.compile(r"^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+([A-Za-z0-9_ &,\-\(\)\/\.\'\’\”\[\]]+)$")

for idx, line in enumerate(content_lines):
    line_str = line.strip()
    if not line_str:
        continue
        
    m = heading_pat.match(line_str)
    if m:
        sec_num_str, title = m.groups()
        sec_num = parse_sec_num(sec_num_str)
        print(f"Line {idx+yaml_lines+1}: Candidate: {sec_num_str} -> {title} (parsed_num: {sec_num})")
    else:
        # If it starts with A. or A.1 etc. but did not match regex, print why
        if re.match(r"^\s*[A-Z](?:\.[0-9]+)*\b", line_str):
            print(f"Line {idx+yaml_lines+1}: Failed regex match: {line_str}")
