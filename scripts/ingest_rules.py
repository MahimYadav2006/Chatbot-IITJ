import os
import sys
import logging
import re
from graphrag.rules_db import RulesDB
from graphrag.rules_parser import RulesParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_rules")

def main():
    db = RulesDB()
    # 1. Clear database
    logger.info("Clearing existing tables in rules.db...")
    db.clear_all()
    
    # 2. Ingest parsed files
    parser = RulesParser()
    files = [
        ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md", "undergraduate", "UG"),
        ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/UG_Curriculum_2022_Scheme_IIT_Jammu.md", "undergraduate_2022", "UG"),
        ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/MTech/M.Tech_RRs___Curric..md", "mtech", "MTech"),
        ("/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/PhD/PhD_RRs.md", "phd", "PhD")
    ]
    
    # Map from (program, source_file, section_number) -> generated section_id
    section_id_map = {}
    
    for filepath, prog_type, program in files:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            continue
            
        logger.info(f"Parsing and ingesting file: {os.path.basename(filepath)} for program={program} (type={prog_type})...")
        sections = parser.parse_file(filepath, prog_type)
        
        for sec in sections:
            db.insert_section(
                section_id=sec["id"],
                section_number=sec["section_number"],
                title=sec["title"],
                full_text=sec["full_text"],
                parent_id=sec["parent_id"],
                program=program,
                source_file=os.path.basename(filepath),
                last_amended=None,
                amendment_note=None
            )
            
            # Map clean section number
            sec_num = sec["section_number"].strip().rstrip(".")
            filename = os.path.basename(filepath)
            section_id_map[(program, filename, sec_num)] = sec["id"]

    # 2b. Ingest standalone amendment documents that do not follow the full
    # numbered-manual structure but override important degree rules.
    amendment_files = [
        (
            "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/Amenedment_in_Rule_2.3.2.2.md",
            "UG",
            "2025-07-15",
        ),
        (
            "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/Amedment_UG_Rules.md",
            "UG",
            None,
        ),
    ]

    for filepath, program, amended_on in amendment_files:
        if not os.path.exists(filepath):
            logger.warning(f"Amendment file not found: {filepath}")
            continue

        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as handle:
            raw_text = handle.read()

        full_text = parser.clean_text(raw_text)
        rule_match = re.search(r"Rule\s+([0-9]+(?:\.[0-9]+)*)", full_text, flags=re.IGNORECASE)
        section_number = rule_match.group(1) if rule_match else "0"

        title = None
        for line in full_text.splitlines():
            line = line.strip()
            if line.lower().startswith("sub:"):
                title = line[4:].strip()
                break
        if not title:
            for line in full_text.splitlines():
                line = line.strip()
                if "open elective" in line.lower():
                    title = line
                    break
        if not title:
            title = f"Amendment in Rule {section_number}"

        stem = re.sub(r"[^a-zA-Z0-9]+", "_", os.path.splitext(filename)[0]).strip("_").lower()
        section_id = f"{program.lower()}_amendment_{stem}"
        db.insert_section(
            section_id=section_id,
            section_number=section_number,
            title=title,
            full_text=full_text,
            parent_id=None,
            program=program,
            source_file=filename,
            last_amended=amended_on,
            amendment_note="Standalone academic rules amendment",
        )
        section_id_map[(program, filename, section_number)] = section_id
            
    # Helper to resolve section_id
    def resolve_sec_id(program: str, filename: str, sec_num: str, fallback_root: str) -> str:
        key = (program, filename, sec_num.strip().rstrip("."))
        if key in section_id_map:
            return section_id_map[key]
        logger.warning(f"Could not resolve section_id for {key}. Using fallback: {fallback_root}")
        return fallback_root

    # 3. Seed Structured Data
    logger.info("Seeding structured tables (grade_scale, credit_requirements, rule_facts, program_milestones)...")
    
    # A. Grade Scale
    grades = [
        ("AA", 10, "Outstanding"),
        ("AB", 9, "Excellent"),
        ("BB", 8, "Very Good"),
        ("BC", 7, "Good"),
        ("CC", 6, "Average"),
        ("CD", 5, "Below Average"),
        ("DD", 4, "Marginal"),
        ("FF", 0, "Poor / Fail"),
        ("SA", 0, "Satisfactory (for CEC)"),
        ("UA", 0, "Unsatisfactory (for CEC)"),
        ("II", 0, "Incomplete"),
        ("PP", 0, "Passed"),
        ("NP", 0, "Not Passed"),
        ("WW", 0, "Withdrawn"),
        ("SS", 0, "Satisfactory (CEC)"),
        ("UU", 0, "Unsatisfactory (CEC)")
    ]
    for grade, gp, desc in grades:
        db.insert_grade(grade, gp, desc)
        
    # B. Credit Requirements
    # UG Requirements
    ug_reqs = [
        ("UG", "IC", "Institute Core", 18.5, None, "Mathematics: 6 credits, Physics: 4.5 credits, Chemistry: 4.5 credits, Biology: 3.5 credits"),
        ("UG", "IE", "Institute Elective", 0.0, None, None),
        ("UG", "DC", "Department Core", 51.0, None, "Range: 51 to 57 credits depending on branch"),
        ("UG", "DE", "Department Elective", 12.0, None, "Range: 12 to 36 credits depending on branch"),
        ("UG", "OE", "Open Elective", 18.0, None, "Includes minimum 6 credits of HSS Open Electives"),
        ("UG", "HSS", "Humanities and Social Sciences Core", 6.0, None, "Additional 9 credits of HSS must be taken as electives / open electives"),
        ("UG", "CGCA", "Co-curricular Activities", 8.0, None, "Bundled into activity units, graded Satisfactory/Unsatisfactory"),
        ("UG", "BTech Project", "B.Tech. Project", 9.0, None, "Graded as 3 + 3 + 3 credits spread over 3 semesters"),
        ("UG", "Minor", "Minor Degree", 12.0, None, "Requires minimum CGPA of 7.0, in addition to regular degree requirements"),
        ("UG", "Specialization", "Specialization", 12.0, None, "Requires minimum CGPA of 7.0, in addition to regular degree requirements"),
        ("UG", "Total Curricular", "Total Curricular Credits", 132.0, None, "Range: 132 to 138 credits depending on branch")
    ]
    for prog, cat, cat_f, min_c, pct, notes in ug_reqs:
        db.insert_credit_requirement(prog, cat, cat_f, min_c, pct, notes)
        
    # C. Rule Facts (Numerical/Constraint grounding)
    facts_raw = [
        # UG facts
        ("threshold", "min_cgpa_minor", "7.0", ">=", "CGPA requirement for Minor and Specialization", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.2.1", "undergraduate_2022_root"),
        ("threshold", "min_cgpa_specialization", "7.0", ">=", "CGPA requirement for Minor and Specialization", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.2.1", "undergraduate_2022_root"),
        ("credit", "min_credits_btech_project", "9.0", "==", "B.Tech project credits", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.3.2.4", "undergraduate_2022_root"),
        ("credit", "min_credits_co_curricular", "8.0", ">=", "Minimum credits required for graduation", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.4", "undergraduate_2022_root"),
        ("credit", "min_credits_minor", "12.0", ">=", "Credits required for minor degree", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.2", "undergraduate_2022_root"),
        ("credit", "min_credits_specialization", "12.0", ">=", "Credits required for specialization", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.2", "undergraduate_2022_root"),
        ("threshold", "max_semester_drop_ug", "1", "<=", "Maximum semester drop allowed during program", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.5", "undergraduate_2022_root"),
        ("threshold", "max_semester_drop_medical_ug", "2", "<=", "Maximum semester drop allowed on severe health issues", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.2.5", "undergraduate_2022_root"),
        ("credit", "max_open_elective_single_department", "9", "<=", "A student may take a maximum of 9 credits from any one department as Open Electives, including one's own department/branch; credits above 9 from a single department shall not be considered in graduating requirements.", "UG", "Amenedment_in_Rule_2.3.2.2.md", "2.3.2.2", "undergraduate_2022_root"),
        ("eligibility", "hss_idp_independent", "independent", "==", "IDP and HSS Open Electives are not clubbed and shall be treated as independent departments for the open-elective credit cap.", "UG", "Amenedment_in_Rule_2.3.2.2.md", "2.3.2.2", "undergraduate_2022_root"),
        ("credit", "min_hss_open_elective_credits", "6", ">=", "A student must take a minimum of 6 credits from the HSS department as Open Electives.", "UG", "Amenedment_in_Rule_2.3.2.2.md", "2.3.2.2", "undergraduate_2022_root"),
        ("credit", "hss_core_requirement", "6", ">=", "Humanities and Social Sciences core requires 6 credits; the listed HSS core courses include Introduction to Literature in English and Introduction to Economics.", "UG", "UG_Curriculum_2022_Scheme_IIT_Jammu.md", "2.3.1", "undergraduate_2022_root"),
        ("policy", "btp_allotment_policy", "section_7.1.1", "==", "UG students choose a BTP topic by the 6th semester; generally 90 percent of admitted batch strength has at least one parent-department supervisor, while 10 percent may choose a sole supervisor outside the parent department, extendable by DUGC.", "UG", "9.5-IIT_Jammu_Rules___Curriculumn.md", "7.1.1", "undergraduate_root"),
        ("policy", "semester_internship_policy", "section_7.2.1", "==", "Semester internship is available in addition to regular internship, subject to credit requirements, TPO guidelines, and prior Dean Academics approval; TPO facilitates/administers the process.", "UG", "9.5-IIT_Jammu_Rules___Curriculumn.md", "7.2.1", "undergraduate_root"),
        ("policy", "course_withdrawal_policy", "section_5.5.2", "==", "Course withdrawal must be completed within the academic-calendar deadline, typically one week after the mid-semester examination; WW is assigned for withdrawal and missing the allowed window may attract FW or UU.", "UG", "9.5-IIT_Jammu_Rules___Curriculumn.md", "5.5.2", "undergraduate_root"),
        ("policy", "department_change_policy", "not_offered", "==", "IIT Jammu has decided not to offer change of department.", "UG", "9.5-IIT_Jammu_Rules___Curriculumn.md", "9", "undergraduate_root"),
        # MTech facts
        ("threshold", "min_cgpa_mtech_course", "6.0", ">=", "CGPA requirement for courses", "MTech", "M.Tech_RRs___Curric..md", "3.5", "mtech_root"),
        ("threshold", "min_cgpa_mtech_dissertation", "6.0", ">=", "CGPA requirement for dissertation", "MTech", "M.Tech_RRs___Curric..md", "3.5", "mtech_root"),
        ("threshold", "pg_diploma_ratio", "0.75", ">=", "Ratio of course credits to apply for PG Diploma if failed M.Tech", "MTech", "M.Tech_RRs___Curric..md", "7.5", "mtech_root"),
        ("policy", "course_withdrawal_policy", "section_6.5.2", "==", "M.Tech course withdrawal is governed by the withdrawal-of-course section in the M.Tech rules.", "MTech", "M.Tech_RRs___Curric..md", "6.5.2", "mtech_root"),
        ("credit", "mtech_ds_is_total_credits", "58", "==", "Total graduation credits for M.Tech Computer Science and Engineering with specializations in Data Sciences (CS-DS) and Information Security (CS-IS)", "MTech", "MTechDSISCreditCorrection.md", "0", "mtech_root"),
        ("credit", "mtech_ds_is_sem1_credits", "15", "==", "Semester I credits for M.Tech Computer Science and Engineering with specializations in Data Sciences (CS-DS) and Information Security (CS-IS)", "MTech", "MTechDSISCreditCorrection.md", "0", "mtech_root"),
        ("credit", "mtech_ds_is_sem2_credits", "17", "==", "Semester II credits for M.Tech Computer Science and Engineering with specializations in Data Sciences (CS-DS) and Information Security (CS-IS) at DA-IICT", "MTech", "MTechDSISCreditCorrection.md", "0", "mtech_root"),
        ("credit", "mtech_ds_is_sem3_credits", "12", "==", "Semester III credits for M.Tech Computer Science and Engineering with specializations in Data Sciences (CS-DS) and Information Security (CS-IS)", "MTech", "MTechDSISCreditCorrection.md", "0", "mtech_root"),
        ("credit", "mtech_ds_is_sem4_credits", "14", "==", "Semester IV credits for M.Tech Computer Science and Engineering with specializations in Data Sciences (CS-DS) and Information Security (CS-IS)", "MTech", "MTechDSISCreditCorrection.md", "0", "mtech_root"),
        # PhD facts
        ("threshold", "min_cgpa_phd_candidacy", "7.0", ">=", "CGPA requirement to register as PhD candidate", "PhD", "PhD_RRs.md", "R.9.2", "phd_root"),
        ("threshold", "max_attempts_comprehensive", "2", "<=", "Maximum attempts to pass comprehensive exam", "PhD", "PhD_RRs.md", "R.8.3", "phd_root"),
        # BTech Minor facts (e.g. Mathematics Minor / general minor rules)
        ("threshold", "min_sgpa_minor_enrollment", "7.0", ">=", "SGPA requirement in all semesters starting from 2nd year to start minor", "UG", "Minor_in_Mathematics.md", "2", "undergraduate_2022_root"),
        ("threshold", "min_sgpa_minor_retention", "7.0", ">=", "SGPA requirement to remain eligible to pursue minor in subsequent semesters", "UG", "Minor_in_Mathematics.md", "2", "undergraduate_2022_root"),
        ("credit", "min_credits_math_minor", "20", ">=", "Minimum credits required from the basket of courses to obtain Minor in Mathematics", "UG", "Minor_in_Mathematics.md", "2", "undergraduate_2022_root"),
        ("threshold", "max_courses_minor_per_semester", "2", "<=", "Maximum number of minor courses a student can opt for in a single semester", "UG", "Minor_in_Mathematics.md", "2", "undergraduate_2022_root"),
        ("threshold", "minor_start_semester", "5", ">=", "Semester from which a student can opt for minor program", "UG", "Minor_in_Mathematics.md", "2", "undergraduate_2022_root")
    ]
    
    for fact_type, fact_key, fact_val, op, cond, prog, filename, sec_num, fallback in facts_raw:
        sec_id = resolve_sec_id(prog, filename, sec_num, fallback)
        db.insert_fact(fact_type, fact_key, fact_val, op, cond, sec_id, prog)
        
    # D. Program Milestones
    milestones_raw = [
        ("PhD", "Course Work Completion", None, "Must complete requisite PhD course work with minimum CGPA of 7.0 prior to comprehensive exam", "PhD_RRs.md", "R.6.3", "phd_root"),
        ("PhD", "Comprehensive Examination (Written & Oral)", "12 to 24 months", "Must complete between 12 and 24 months from joining, max 2 attempts", "PhD_RRs.md", "R.8.4", "phd_root"),
        ("PhD", "Research Plan Seminar & SOTA Presentation", "Within 3 months of passing written exam", "Oral exam and research plan proposal presentation", "PhD_RRs.md", "R.8.1", "phd_root"),
        ("PhD", "Formal Candidacy Registration", "Soon after passing comprehensive exam", "Requires coursework CGPA >= 7.0 and approved research plan", "PhD_RRs.md", "R.9.2", "phd_root")
    ]
    for prog, ms, deadline, details, filename, sec_num, fallback in milestones_raw:
        sec_id = resolve_sec_id(prog, filename, sec_num, fallback)
        db.insert_program_milestone(prog, ms, deadline, details, sec_id)
        
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
