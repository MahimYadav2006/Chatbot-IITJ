"""Unit tests for crawler URL normalization and department resolution."""

import os

import departments
from departments import get_department, get_markdown_dir, resolve_department_code
from utils import canonicalize_url, classify_discovered_url, is_generic_page


BASE_URL = "https://iitjammu.ac.in/computer_science_engineering"


def test_resolve_department_alias():
    assert resolve_department_code("cse") == "computer_science_engineering"
    assert get_department("cse")["base_url"] == BASE_URL


def test_markdown_dir_prefers_scraped_data_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(departments, "SCRAPED_DATA_ROOT", str(tmp_path / "scraped_data"))
    target = tmp_path / "scraped_data" / "computer_science_engineering"
    os.makedirs(target, exist_ok=True)
    assert get_markdown_dir("cse") == str(target)


def test_canonicalize_root_relative_department_href_from_nested_page():
    current_url = f"{BASE_URL}/program-list"
    href = "computer_science_engineering/program-list/ug-programme"
    assert canonicalize_url(BASE_URL, current_url, href) == f"{BASE_URL}/program-list/ug-programme"


def test_canonicalize_tilde_profile_href_from_listing_page():
    current_url = f"{BASE_URL}/faculty-list"
    href = "~aroofaimen"
    assert canonicalize_url(BASE_URL, current_url, href) == f"{BASE_URL}/faculty-list/~aroofaimen"


def test_canonicalize_strips_nonbreaking_space_and_index_html():
    current_url = BASE_URL
    href = "message-from-deparment-hod\xa0"
    assert canonicalize_url(BASE_URL, current_url, href) == f"{BASE_URL}/message-from-deparment-hod"
    assert canonicalize_url(BASE_URL, current_url, f"{BASE_URL}/index.html") == BASE_URL


def test_classify_department_page_and_binary():
    page_kind, page_reason = classify_discovered_url(f"{BASE_URL}/about-us", BASE_URL, "iitjammu.ac.in")
    binary_kind, binary_reason = classify_discovered_url(
        "https://iitjammu.ac.in/faculty/documents/cv_123.pdf",
        BASE_URL,
        "iitjammu.ac.in",
    )
    reject_kind, reject_reason = classify_discovered_url("https://lms.iitjammu.ac.in", BASE_URL, "iitjammu.ac.in")

    assert (page_kind, page_reason) == ("page", "department-page")
    assert (binary_kind, binary_reason) == ("binary", "binary-file")
    assert (reject_kind, reject_reason) == ("reject", "external-domain")


def test_generic_page_detection():
    assert is_generic_page(
        "Indian Institute of Technology Jammu | Leading Engineering Institute for Future Innovators",
        "Computer Science & Engineering Copyright © 2020 IIT Jammu, all rights reserved",
    )


def test_section_url_matching():
    # Test www. vs non-www domain classification
    base_url_www = "https://www.iitjammu.ac.in/academics"
    allowed_domain_www = "www.iitjammu.ac.in"

    # 1. Matches when domain is non-www in candidate but www in allowed_domain
    candidate_non_www = "https://iitjammu.ac.in/academics/program-list"
    kind1, reason1 = classify_discovered_url(candidate_non_www, base_url_www, allowed_domain_www)
    assert kind1 == "page"
    assert reason1 == "department-page"

    # 2. Matches when domain is www in candidate but non-www in base_url
    base_url_non_www = "https://iitjammu.ac.in/alumni-affairs"
    allowed_domain_non_www = "iitjammu.ac.in"
    candidate_www = "https://www.iitjammu.ac.in/alumni-affairs/events"
    kind2, reason2 = classify_discovered_url(candidate_www, base_url_non_www, allowed_domain_non_www)
    assert kind2 == "page"
    assert reason2 == "department-page"

    # 3. Google Sites paths
    sites_base = "https://sites.google.com/iitjammu.ac.in/hindicell"
    sites_allowed = "sites.google.com"
    sites_candidate = "https://sites.google.com/iitjammu.ac.in/hindicell/announcements"
    kind3, reason3 = classify_discovered_url(sites_candidate, sites_base, sites_allowed)
    assert kind3 == "page"
    assert reason3 == "department-page"

    # 4. Google Sites path out of scope
    sites_other = "https://sites.google.com/iitjammu.ac.in/tlu"
    kind4, reason4 = classify_discovered_url(sites_other, sites_base, sites_allowed)
    assert kind4 == "reject"
    assert reason4 == "outside-department-scope"

