import re

files = [
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/PhD/PhD_RRs.md", "phd")
]

for filepath, prog in files:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip YAML header
    yaml_lines = 0
    if lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                yaml_lines = i + 1
                break

    content_lines = lines[yaml_lines:]

    def parse_sec_num(s):
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
                if len(p) == 1 and p.isalpha():
                    parts.append(ord(p.upper()) - ord("A") + 1)
                else:
                    return None
        return tuple(parts)

    heading_pat = re.compile(r"^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+([A-Za-z0-9_ &,\-\(\)\/\.\'\’\”\[\]]+)$")

    active_path = []
    detected_headings = []
    body_started = False
    
    use_mixed_case = prog in ("undergraduate_2022", "phd")

    for idx, line in enumerate(content_lines):
        line_str = line.strip()
        if not line_str:
            continue
        
        if "..." in line_str or ".. " in line_str or " . ." in line_str or line_str.endswith(" ee") or ("]" in line_str and any(char.isdigit() for char in line_str[-5:])):
            continue
            
        m = heading_pat.match(line_str)
        if m:
            sec_num_str, title = m.groups()
            title = title.strip().rstrip("]").rstrip(")").strip()
            
            if title.endswith(".") or title.endswith(";") or title.endswith(",") or title.endswith(":") or title.endswith("?"):
                continue
            if len(title) > 90 or len(title) < 2:
                continue
                
            sec_num = parse_sec_num(sec_num_str)
            if not sec_num:
                continue
                
            # LEVEL 1 HEADING RULES
            if len(sec_num) == 1 or (len(sec_num) == 2 and sec_num[0] == 18):
                if prog == "phd" and len(sec_num) == 1:
                    # Must be alphabetic (A, B, C...)
                    if not sec_num_str.rstrip(".")[0].isalpha():
                        continue
                if use_mixed_case:
                    # Mixed case allowed, but must be short (<= 6 words) and capitalized
                    if len(title.split()) > 6 or not title[0].isupper():
                        continue
                else:
                    # Must be UPPERCASE
                    letters = [c for c in title if c.isalpha()]
                    if letters:
                        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                        if upper_ratio < 0.7:
                            continue
            
            # Start body on A (which is ord("A") - ord("A") + 1 = 1)
            if not body_started:
                if sec_num == (1,):
                    body_started = True
                    active_path = [sec_num]
                    detected_headings.append((idx + yaml_lines + 1, sec_num_str, title))
                continue
                
            is_valid = False
            
            # Auto-initialize regulations namespace if needed
            if sec_num[0] == 18 and not any(p[0] == 18 for p in active_path):
                active_path = [(18,)]
                
            if active_path:
                current = active_path[-1]
                
                if len(sec_num) == len(current) + 1 and sec_num[:-1] == current:
                    is_valid = True
                elif len(sec_num) <= len(current) and len(sec_num) - 1 < len(active_path):
                    ancestor = active_path[len(sec_num)-1]
                    if len(sec_num) == 1:
                        if sec_num[0] == active_path[0][0] + 1:
                            is_valid = True
                    elif len(sec_num) == 2 and sec_num[0] == 18:
                        # Regulations sibling transition
                        if len(ancestor) >= 2:
                            if sec_num[1] > ancestor[1]:
                                is_valid = True
                        else:
                            is_valid = True
                    else:
                        if sec_num[-1] > ancestor[-1] and sec_num[:-1] == ancestor[:-1]:
                            is_valid = True
                    
            if is_valid:
                active_path = [p for p in active_path if len(p) < len(sec_num)]
                active_path.append(sec_num)
                detected_headings.append((idx + yaml_lines + 1, sec_num_str, title))

    print("=" * 60)
    print(f"File: {filepath.split('/')[-1]}")
    print(f"Detected {len(detected_headings)} valid headings:")
    for idx, sec_num_str, title in detected_headings:
        print(f"  Line {idx}: {sec_num_str} -> {title}")
