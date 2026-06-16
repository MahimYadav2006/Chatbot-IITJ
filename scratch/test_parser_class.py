from graphrag.rules_parser import RulesParser

parser = RulesParser()

files = [
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md", "undergraduate"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/UG_Curriculum_2022_Scheme_IIT_Jammu.md", "undergraduate_2022"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/MTech/M.Tech_RRs___Curric..md", "mtech"),
    ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/PhD/PhD_RRs.md", "phd")
]

for filepath, prog in files:
    sections = parser.parse_file(filepath, prog)
    print("=" * 60)
    print(f"File: {filepath.split('/')[-1]} -> {prog}")
    print(f"Parsed {len(sections)} sections:")
    for sec in sections[:5]:
        print(f"  ID: {sec['id']}, Num: {sec['section_number']}, Title: {sec['title']}, Parent: {sec['parent_id']}")
        print(f"    Preview: {repr(sec['full_text'][:70])}")
