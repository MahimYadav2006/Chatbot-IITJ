"""
Central department registry for IIT Jammu multi-department chatbot.
All department-specific configuration lives here.
"""
import os
from env_config import load_env_file

load_env_file()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DATA_ROOT = os.environ.get(
    "SCRAPED_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "scraped_data"),
)

DEPARTMENTS = {
    "administration": {
        "name": "Administration",
        "full_name": "Administration of IIT Jammu",
        "base_url": "https://iitjammu.ac.in",
        "template": "A",
    },
    "ee": {
        "name": "Electrical Engineering",
        "full_name": "Department of Electrical Engineering",
        "base_url": "https://iitjammu.ac.in/ee",
        "template": "A",
        "official_contact_email": "hod.ee@iitjammu.ac.in",
    },
    "computer_science_engineering": {
        "name": "Computer Science & Engineering",
        "full_name": "Department of Computer Science and Engineering",
        "base_url": "https://iitjammu.ac.in/computer_science_engineering",
        "template": "B",
        "aliases": ["cse"],
    },
    "mechanical_engineering": {
        "name": "Mechanical Engineering",
        "full_name": "Department of Mechanical Engineering",
        "base_url": "https://iitjammu.ac.in/mechanical_engineering",
        "template": "A",
    },
    "civil_engineering": {
        "name": "Civil Engineering",
        "full_name": "Department of Civil Engineering",
        "base_url": "https://iitjammu.ac.in/civil_engineering",
        "template": "B",
    },
    "chemical-engineering": {
        "name": "Chemical Engineering",
        "full_name": "Department of Chemical Engineering",
        "base_url": "https://iitjammu.ac.in/chemical-engineering",
        "template": "B",
        "aliases": ["chemical"],
    },
    "bsbe": {
        "name": "Biosciences & Bioengineering",
        "full_name": "Department of Biosciences and Bioengineering",
        "base_url": "https://iitjammu.ac.in/bsbe",
        "template": "A",
    },
    "chemistry": {
        "name": "Chemistry",
        "full_name": "Department of Chemistry",
        "base_url": "https://iitjammu.ac.in/chemistry",
        "template": "B",
    },
    "hss": {
        "name": "Humanities & Social Sciences",
        "full_name": "Department of Humanities and Social Sciences",
        "base_url": "https://iitjammu.ac.in/hss",
        "template": "B",
    },
    "idp": {
        "name": "Interdisciplinary Programmes",
        "full_name": "Interdisciplinary Programmes",
        "base_url": "https://iitjammu.ac.in/idp",
        "template": "B",
    },
    "materials-engineering": {
        "name": "Materials Engineering",
        "full_name": "Department of Materials Engineering",
        "base_url": "https://iitjammu.ac.in/materials-engineering",
        "template": "B",
        "aliases": ["materials"],
    },
    "mathematics": {
        "name": "Mathematics",
        "full_name": "Department of Mathematics",
        "base_url": "https://iitjammu.ac.in/mathematics",
        "template": "B",
    },
    "physics": {
        "name": "Physics",
        "full_name": "Department of Physics",
        "base_url": "https://iitjammu.ac.in/physics",
        "template": "B",
    },
}

CORRECT_LABS = {
    "mechanical_engineering": [
        "Fluid Mechanics Lab",
        "Control Engineering Lab",
        "Kinematics & Dynamics of Machine Lab",
        "Heat & Mass Transfer Lab",
        "Energy Systems Lab",
        "Solid Mechanics Lab",
        "Manufacturing Lab",
    ],
    "ee": [
        "Low Voltage Lab 2",
        "Low Voltage Lab1",
        "Prototype Design and Development Lab",
        "Undergraduate and Postgraduate Labs",
        "Underwater Research Lab",
        "Underwater Research lab",
        "High Voltage Lab",
        "IC Reliabality, Security & Quality Laboratory",
        "AADHRIT Lab",
    ],
    "chemical-engineering": [
        "Fluid Flow & Mechanical Operations Laboratory",
        "Heat Transfer & Thermodynamics Laboratory",
        "Process Instrumentation Dynamics & Control Lab",
        "Process Modelling & Simulation Lab",
        "Separation Processes And Chemical Reaction Engineering Laboratory",
        "Research Laboratories : Process Intensification and Nanoscale Advanced Materials (Faculty Name - Dr. Gaurav A. Bhaduri) , Microfluidics and Energy Systems Lab (Faculty Name - Dr. Ravi K. Arun) , Microfludics Design and Bioengineering Lab - (Faculty Name - Dr. Dharitri Rath)",
    ],
    "physics": [
        "Material Research Laboratory (MRL)",
        "Solar Research Lab (SRL)",
        "Shivalik Plasma Laboratory",
        "Optoelectronics and Device Physics Laboratory",
    ],
    "chemistry": [
        "Chemistry Laboratory",
    ],
    "bsbe": [
        "UG Bio Lab",
        "Genetic Engineering and Tissue culture lab",
        "Nanodiagnostics and Therapeutics Lab",
    ],
    "hss": [
        "Language Experiential Lab",
    ],
    "civil_engineering": [
        "Environmental Engineering Laboratory",
        "Geo-STRIDE Lab",
        "Fluid Mechanics lab",
        "Rock Mechanics and Geology Lab",
        "Soil Mechanics Lab",
        "Water Resources Lab",
    ],
}

DEPARTMENT_ALIASES = {
    alias: code
    for code, config in DEPARTMENTS.items()
    for alias in config.get("aliases", [])
}


def resolve_department_code(code: str) -> str:
    """Resolve a user-facing department code or alias to the canonical code."""
    normalized = (code or "").strip().lower()
    if normalized in DEPARTMENTS:
        return normalized
    if normalized in DEPARTMENT_ALIASES:
        return DEPARTMENT_ALIASES[normalized]
    raise KeyError(f"Department code '{code}' not found in registry.")

def get_department(code: str) -> dict:
    """Get department config by code. Raises KeyError if not found."""
    return DEPARTMENTS[resolve_department_code(code)]

def get_all_codes() -> list:
    """Get a list of all department codes."""
    return list(DEPARTMENTS.keys())

def get_scraped_data_root() -> str:
    """Return the root folder that stores crawled markdown for all departments."""
    return SCRAPED_DATA_ROOT

def get_scraped_markdown_dir(code: str) -> str:
    """Return the canonical crawl output directory under `scraped_data/`."""
    canonical = resolve_department_code(code)
    return os.path.join(SCRAPED_DATA_ROOT, canonical)

def get_legacy_markdown_dir(code: str) -> str:
    """Return the original checked-in markdown directory path for a department."""
    canonical = resolve_department_code(code)
    return os.path.join(PROJECT_ROOT, f"iitjammu_{canonical}_markdown")

def get_markdown_dir(code: str) -> str:
    """Return the markdown directory path for a department.

    New crawls are stored under `scraped_data/<department>/`.
    For backward compatibility, if that folder does not exist yet but the legacy
    checked-in markdown directory exists, fall back to the legacy location.
    """
    canonical = resolve_department_code(code)
    scraped_dir = get_scraped_markdown_dir(canonical)
    legacy_dir = get_legacy_markdown_dir(canonical)
    if os.path.exists(scraped_dir):
        return scraped_dir
    if os.path.exists(legacy_dir):
        return legacy_dir
    return scraped_dir

def get_data_dir(code: str) -> str:
    """Return the data directory path for a department."""
    canonical = resolve_department_code(code)
    return os.path.join(PROJECT_ROOT, "data", canonical)

SECTIONS = {
    "academics": {"name": "Academics", "base_url": "https://www.iitjammu.ac.in/academics"},
    "alumni-affairs": {"name": "Alumni Affairs", "base_url": "https://iitjammu.ac.in/alumni-affairs"},
    "cds": {"name": "Career Development Services (CDS)", "base_url": "https://iitjammu.ac.in/cds"},
    "counselling": {"name": "Counselling", "base_url": "https://iitjammu.ac.in/counselling"},
    "di": {"name": "DI", "base_url": "https://iitjammu.ac.in/di"},
    "e2": {"name": "E2", "base_url": "https://www.iitjammu.ac.in/e2"},
    "saral": {"name": "SARAL (Student Portal)", "base_url": "https://iitjammu.ac.in/saral"},
    "accounts": {"name": "Accounts", "base_url": "https://www.iitjammu.ac.in/accounts"},
    "hindicell": {"name": "Hindi Cell", "base_url": "https://sites.google.com/iitjammu.ac.in/hindicell"},
    "ir": {"name": "International Relations (IR)", "base_url": "https://ir.iitjammu.ac.in/"},
    "library": {"name": "Library", "base_url": "https://library.iitjammu.ac.in/"},
    "medical-centre": {"name": "Medical Centre", "base_url": "https://iitjammu.ac.in/medical-centre/"},
    "osd": {"name": "Outreach & Skill Development (OSD)", "base_url": "https://iitjammu.ac.in/osd/"},
    "sp": {"name": "Stores & Purchase (Procurement)", "base_url": "https://www.iitjammu.ac.in/sp"},
    "rc": {"name": "Research Centre (RC)", "base_url": "https://www.iitjammu.ac.in/rc"},
    "sw": {"name": "Student Welfare (SW)", "base_url": "https://iitjammu.ac.in/sw"},
    "security": {"name": "Security Section", "base_url": "https://www.iitjammu.ac.in/security/"},
    "tlu": {"name": "Teaching Learning Unit (TLU)", "base_url": "https://sites.google.com/iitjammu.ac.in/tlu"},
    "tinkerers-lab": {"name": "Tinkerers' Lab", "base_url": "https://iitjammu.ac.in/tinkerers-lab"},
    # ── Student Data Sections ────────────────────────────────────────────
    "students-faq": {"name": "Student FAQ", "base_url": "https://iitjammu.ac.in/faq-main-website"},
    "students-schedule": {"name": "Academic Calendar & Schedules", "base_url": "https://iitjammu.ac.in/calendar-schedule-time-table"},
    "students-phd-admissions": {"name": "PhD Admissions", "base_url": "https://iitjammu.ac.in/phd"},
    "students-pg-admissions": {"name": "PG Admissions", "base_url": "https://iitjammu.ac.in/pg-admissions"},
    "students-ug-admissions": {"name": "UG Admissions", "base_url": "https://iitjammu.ac.in/ugadm"},
    "students-certificate-programs": {"name": "Certificate Programs", "base_url": "https://iitjammu.ac.in/certificate-programs"},
    "students-online-education": {"name": "Online Education", "base_url": "https://iitjammu.ac.in/online-education"},
    "students-pmrf": {"name": "PMRF Scheme", "base_url": "https://iitjammu.ac.in/pmrf"},
    "students-visvesvaraya": {"name": "Visvesvaraya PhD Scheme", "base_url": "https://iitjammu.ac.in/visvesvaraya-phd"},
    "students-why-iitjammu": {"name": "Why IIT Jammu", "base_url": "https://www.iitjammu.ac.in/why-iitjammu"},
    "students-academic-downloads": {"name": "Academic General Downloads", "base_url": "https://iitjammu.ac.in/academics/academics-general-downloads"},
    # ── Media Section ────────────────────────────────────────────
    "media": {"name": "Media & Communications", "base_url": "https://iitjammu.ac.in/events"},
    # ── Quick Links Sections ─────────────────────────────────────
    "quick-adjunct-faculty": {"name": "Adjunct Faculty", "base_url": "https://iitjammu.ac.in/adjunct-faculty"},
    "quick-anti-ragging": {"name": "Anti-Ragging", "base_url": "https://iitjammu.ac.in/anti-ragging"},
    "quick-committees": {"name": "Institute Committees", "base_url": "https://iitjammu.ac.in/equal-opportunity-cell"},
    "quick-staff": {"name": "Institute Staff Directory", "base_url": "https://iitjammu.ac.in/staff-page"},
    "quick-contacts": {"name": "Contact Directory", "base_url": "https://iitjammu.ac.in/voip-directory"},
    "quick-rti": {"name": "RTI & Transparency", "base_url": "https://iitjammu.ac.in/rti"},
}

# Mapping from student section codes to their actual scraped_data subdirectory
# (they live under scraped_data/students/ instead of scraped_data/sections/)
STUDENT_SECTION_FOLDER_MAP = {
    "students-faq": "students/faq-main-website",
    "students-schedule": "students/calendar-schedule-time-table",
    "students-phd-admissions": "students/phd",
    "students-pg-admissions": "students/pg-admissions",
    "students-ug-admissions": "students/ugadm",
    "students-certificate-programs": "students/certificate-programs",
    "students-online-education": "students/online-education",
    "students-pmrf": "students/pmrf",
    "students-visvesvaraya": "students/visvesvaraya-phd",
    "students-why-iitjammu": "students/why-iitjammu",
    "students-academic-downloads": "students/academics-general-downloads",
}

# Mapping from media section code to Media directory
MEDIA_SECTION_FOLDER_MAP = {
    "media": "Media",
}

# Mapping from quick section codes to their actual scraped_data subdirectory
QUICK_SECTION_FOLDER_MAP = {
    "quick-adjunct-faculty": "Quick/adjunct-faculty",
    "quick-anti-ragging": "Quick/anti-ragging",
    "quick-committees": "Quick",         # Multi-source: pulls from multiple Quick subdirs
    "quick-staff": "Quick/staff-page",
    "quick-contacts": "Quick",            # Multi-source: voip-directory + welcome-contacts
    "quick-rti": "Quick/rti",
}

# For quick sections that pull from multiple subdirectories, list them here.
# The builder will iterate over all source dirs for these sections.
QUICK_MULTI_SOURCE = {
    "quick-committees": [
        "Quick/equal-opportunity-cell",
        "Quick/institute-ethics-committee",
        "Quick/internal-complaint-committee",
        "Quick/st-sc-cell",
        "Quick/Pdf-data",
    ],
    "quick-contacts": [
        "Quick/voip-directory",
        "Quick/welcome-contacts",
    ],
    "quick-anti-ragging": [
        "Quick/anti-ragging",
        "Quick/Pdf-data",
    ],
    "quick-rti": [
        "Quick/rti",
        "Quick/suo-moto-disclosure",
    ],
}

# Boilerplate filenames to skip during student section ingestion
STUDENT_SECTION_SKIP_PATTERNS = [
    "_nirf_", "_rti_", "_SGRC_", "_index.md",
    "nirf_2026", "CERT-IN_Certificate",
    "Student_Grievance_Redressal_Committee",
]

MEDIA_SECTION_SKIP_PATTERNS = [
    "_index.md", "crawl_manifest.json",
    "CERT-IN", "SGRC", "nirf_2026",
]

QUICK_SECTION_SKIP_PATTERNS = [
    "_index.md", "crawl_manifest.json",
    "00_combined_",
]


def get_section_markdown_dir(code: str) -> str:
    """Return the canonical crawl output directory for a section.

    Student sections live under ``scraped_data/students/``.
    Quick sections live under ``scraped_data/Quick/``.
    Institutional sections live under ``scraped_data/sections/``.
    """
    if code not in SECTIONS:
        raise KeyError(f"Section code '{code}' not found in registry.")
    if code in STUDENT_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, STUDENT_SECTION_FOLDER_MAP[code])
    if code in MEDIA_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, MEDIA_SECTION_FOLDER_MAP[code])
    if code in QUICK_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, QUICK_SECTION_FOLDER_MAP[code])
    return os.path.join(SCRAPED_DATA_ROOT, "sections", code)


def get_section_data_dir(code: str) -> str:
    """Return the data directory path for a section."""
    if code not in SECTIONS:
        raise KeyError(f"Section code '{code}' not found in registry.")
    return os.path.join(PROJECT_ROOT, "data", "sections", code)

