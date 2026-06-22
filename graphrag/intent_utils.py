import re


def _normalized_query(query: str) -> str:
    """Normalize common OCR/user typos before lightweight intent checks."""
    q = re.sub(r"\s+", " ", query.lower()).strip()
    replacements = {
        "tthe": "the",
        "compulsoy": "compulsory",
        "compulsary": "compulsory",
        "withdrawl": "withdrawal",
        "withdrwal": "withdrawal",
        "divison": "division",
        "divisons": "divisions",
        "recognitions": "recognition",
        "evalution": "evaluation",
        "btp": "btech project",
        "b.tech project": "btech project",
        "b.tech. project": "btech project",
        "oe": "open elective",
        "oes": "open electives",
        "changing": "change",
    }
    for old, new in replacements.items():
        q = re.sub(rf"\b{re.escape(old)}\b", new, q)
    return q


def is_academic_rules_query(query: str) -> bool:
    """Detect if a query is about academic rules, regulations, or policies.
    
    Uses precise keywords and patterns, and includes guards to prevent false matches
    on queries meant for other departments or sections (e.g. CDS, Counselling, specific faculty).
    """
    q = _normalized_query(query)
    
    # 1. Negative guards: if these are present, route to their respective sections/departments instead
    cds_indicators = ("placement", "placements", "recruit", "recruiting", "recruiter", "recruiters", "job", "jobs", "salary", "salary package", "highest package", "average package", "ctc", "lpa", "companies visiting", "placement policy", "companies")
    counselling_indicators = ("counselling", "counseling", "stress", "anxiety", "depression", "mental health", "counselor", "counsellor", "therapy")
    medical_indicators = ("medical", "health centre", "doctor", "doctors", "specialist", "dentist", "physiotherapist", "ambulance", "pharmacy", "hospital", "hospitals", "empaneled", "cghs")
    alumni_indicators = ("alumni", "gold medal", "silver medal", "convocation award", "medalist", "medalists")
    sports_hostel_fests = ("sports", "gym", "cricket", "basketball", "hostel", "hostels", "canary", "braeg", "fulgar", "dedhar", "egret", "fest", "fests", "festival", "festivals", "anhad", "pravaah", "nexus")
    
    # Also guard against generic faculty research or department queries
    faculty_research_indicators = ("research interest", "research area", "research work", "publication", "publications", "paper", "papers", "journal", "conference", "patent", "patents", "startup", "startups", "teaches", "teaching", "who works on", "who is working on", "who researches")
    
    # Check if the query is about the rules or criteria of getting medals/awards
    medal_rules_query = any(m in q for m in ("gold medal", "silver medal", "convocation award")) and any(r in q for r in ("how", "get", "criteria", "eligibility", "rule", "rules", "regulation", "regulations", "procedure", "requirement", "requirements", "obtain", "win", "receive", "eligible"))

    if any(term in q for term in cds_indicators):
        return False
    if any(term in q for term in counselling_indicators):
        return False
    if any(term in q for term in medical_indicators):
        return False
    if medal_rules_query:
        return True
    if any(term in q for term in alumni_indicators):
        return False
    if any(term in q for term in sports_hostel_fests):
        return False
    if any(term in q for term in faculty_research_indicators):
        return False

    # Guard against identity queries (e.g. "Who is Prakriti Gupta")
    # If the query contains a name honorific and doesn't explicitly ask for rules, let it go to identity routing
    if re.search(r"\b(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?)\s+\w+", q):
        # Unless they explicitly ask about a rule relating to that person, e.g. "guidelines for Prof X"
        if not any(term in q for term in ("rule", "rules", "regulation", "regulations", "policy", "policies")):
            return False

    # 2. Positive indicators
    rules_keywords = (
        "rule", "rules", "regulation", "regulations", "policy", "policies", 
        "manual", "handbook", "statute", "statutes", "guideline", "guidelines", 
        "ordinance", "ordinances", "requirement", "requirements", "criteria", 
        "eligibility"
    )
    
    malpractice_keywords = (
        "proxy", "identical answer", "identical answers", "unfair means", 
        "cheating", "malpractice", "malpractices", "deterrence", "plagiarism", 
        "copying", "disciplining", "disciplinary"
    )
    
    grade_keywords = (
        "cgpa", "sgpa", "gpa", "conversion", "convert", "percentage", 
        "division", "divisions", "grade point", "grade points", "grading scale", 
        "fail grade", "pointer", "pointers", "backlog", "backlogs", "grade card",
        "transcript", "transcripts"
    )
    
    jrf_srf_keywords = (
        "jrf", "srf", "stipend", "fellowship", "contingency", "scholarship", 
        "assistantship", "gate score", "gate qualified"
    )
    
    milestone_keywords = (
        "comprehensive exam", "comprehensive examination", "candidacy", 
        "synopsis", "sota", "state of the art", "seminar", "seminars", 
        "timeline", "deadlines", "semester drop", "drop a semester", 
        "academic milestone", "milestones", "credit requirement", 
        "credit requirements", "credits required", "minimum credit", 
        "minimum credits", "course registration", "add drop", "late registration",
        "credit limit", "maximum credits", "course load", "academic load",
        "course work", "coursework", "phd registration", "thesis submission",
        "dissertation", "minor programme", "specialization requirement",
        "ug diploma", "pg diploma", "degree requirement", "degree requirements",
        "minor program", "minor degree", "minor in", "minors in", "minnor in", "minnors in", "specialization in",
        "curriculum", "course structure", "course offering", "elective basket",
        "elective bucket", "elective track", "list of courses", "course syllabus",
        "course code", "course name", "ltp", "ltp structure", "credits of",
        "credit distribution", "credits do", "credits to graduate", "credits needed"
    )

    academic_policy_phrases = (
        "withdrawal of course", "withdrawal of courses", "withdraw course",
        "course withdrawal", "drop course", "drop courses",
        "evaluation scheme", "evaluation schemes", "evaluation mode",
        "evaluation modes", "theoretical course", "theoretical courses",
        "laboratory course", "laboratory courses",
        "award recognition", "awards recognition", "awards and recognition",
        "awards and recognitions", "kind of recognition", "kinds of recognition",
        "different kind of recognition", "different kinds of recognition",
        "institute gold medal", "institute silver medal",
        "course structure for students under ra category",
        "course structure for students under ta category",
        "course structure under ra category", "course structure under ta category",
        "ra category", "ta category", "sp category", "ta/sp category",
        "course code convention", "course coded as", "coded as",
        "btech project allotment", "project allotment", "btp allotment",
        "semester internship", "provision of semester internship",
        "open elective", "open electives", "maximum credits", "single department",
        "hss open elective", "hss open electives", "idp and hss", "hss and idp",
        "minimum degree requirement", "minimum degree requirements",
        "compulsory hss", "hss course", "hss courses",
        "change of department", "department change", "branch change",
        "change department", "change branch", "change of branch",
    )
    
    # Check if any strong academic rule keyword is present
    notification_keywords = (
        "dpgc", "dugc", "committee member", "committee chairperson",
        "faculty advisor", "programme coordinator", "program coordinator",
        "fee structure", "fee notification", "tuition fee", "fee waiver",
        "euler", "earn while you learn", "travel grant", "travel support",
        "contingency grant", "research contingency", "bs honours", "bs honors",
        "open research day", "stic dinner", "summer term incentive",
        "htra", "partial financial support", "international student fee",
        "study in india", "sii", "early start phd"
    )
    if any(term in q for term in notification_keywords):
        return True

    if any(term in q for term in jrf_srf_keywords):
        return True
    if any(term in q for term in malpractice_keywords):
        return True
    
    # For conversion/grade terms, we want to make sure it's academic, e.g. "cgpa to percentage" or "cgpa conversion"
    if any(term in q for term in grade_keywords):
        # Guard: if it's just "grade" or "gpa", make sure it's in an academic context or has other rule terms
        if any(term in q for term in ("conversion", "convert", "percentage", "rules", "rule", "policy", "scale", "point")):
            return True
        if "cgpa" in q or "sgpa" in q:
            return True
            
    if any(term in q for term in milestone_keywords):
        return True

    if any(term in q for term in academic_policy_phrases):
        return True

    if re.search(r"\b(?:[a-z]\s*)?[a-z]{2,3}\s*\d{3}\s*[up]\s*\d\s*[meix]\b", q):
        return True
        
    # Check if a general rules keyword is combined with academic terms
    academic_context_terms = (
        "academic", "academics", "course", "courses", "class", "classes", 
        "attendance", "attend", "exam", "exams", "examination", "examinations", 
        "student", "students", "faculty", "instructor", "professor", "programme", 
        "programmes", "program", "programs", "btech", "b.tech", "mtech", "m.tech", 
        "phd", "ph.d", "doctoral", "undergraduate", "postgraduate", "ug", "pg", 
        "semester", "semesters", "degree", "degrees"
    )
    if any(term in q for term in rules_keywords):
        if any(term in q for term in academic_context_terms):
            return True

    return False
