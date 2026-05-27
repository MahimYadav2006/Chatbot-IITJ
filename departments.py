"""
Central department registry for IIT Jammu multi-department chatbot.
All department-specific configuration lives here.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DATA_ROOT = os.environ.get(
    "SCRAPED_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "scraped_data"),
)

DEPARTMENTS = {
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
