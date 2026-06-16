import re
from graphrag.rules_parser import RulesParser

parser = RulesParser()
line = "2.2.2 Minor and Specialization in B.Tech."
print("Line matches num_pat:", bool(parser.num_pat.match(line)))
m = parser.num_pat.match(line)
if m:
    sec_num_str, title_str = m.groups()
    print("sec_num_str:", sec_num_str)
    print("title_str:", title_str)
    title_str = title_str.strip().rstrip("]").rstrip(")").strip()
    is_abbreviation = any(title_str.endswith(abbr) for abbr in ("B.Tech.", "M.Tech.", "Ph.D.", "B.T.", "M.T.", "Dr.", "Prof.", "i.e.", "e.g.", "etc."))
    print("is_abbreviation:", is_abbreviation)
    print("ends with dot:", title_str.endswith("."))
    print("is rejected:", (title_str.endswith(".") and not is_abbreviation))
