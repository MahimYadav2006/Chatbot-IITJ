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
