import os
import re
import requests
import csv
import io
import tempfile
import uuid
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment
import pdfplumber
import docx
import openpyxl
from PIL import Image
import pytesseract
from playwright.sync_api import sync_playwright

# --- Configuration ---
BASE_URL = "https://iitjammu.ac.in/ee"
OUTPUT_DIR = "iitjammu_ee_markdown"
ALLOWED_PREFIX = "https://iitjammu.ac.in/ee"
ALLOWED_DOMAIN = "iitjammu.ac.in"

def discover_links(base_url, context):
    """Crawl the EE department site and collect all internal links using Playwright browser context."""
    all_links = set()
    pages_to_visit = {base_url + "/index.html", base_url + "/"}
    visited_pages = set()
    
    while pages_to_visit:
        current_url = pages_to_visit.pop()
        if current_url in visited_pages:
            continue
            
        try:
            print(f"Discovering links from: {current_url}")
            page = context.new_page()
            # Set a generous timeout and wait until the network is idle
            page.goto(current_url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000) # Give dynamic components a moment to mount
            
            visited_pages.add(current_url)
            html_content = page.content()
            page.close()
            
            soup = BeautifulSoup(html_content, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urljoin(current_url, href)
                
                # Keep only EE section pages or allowed file types from our domain
                if full_url.startswith(ALLOWED_PREFIX) or (
                    ALLOWED_DOMAIN in full_url and
                    any(full_url.lower().endswith(ext) for ext in [".pdf", ".xlsx", ".docx", ".csv", ".jpg", ".png", ".gif"])
                ):
                    # Remove URL fragment identifier (hash section)
                    parsed_full = urlparse(full_url)
                    normalized_url = parsed_full._replace(fragment="").geturl()
                    
                    if normalized_url not in all_links:
                        all_links.add(normalized_url)
                        # If it's an HTML page, crawl it further
                        if normalized_url.startswith(ALLOWED_PREFIX) and not any(
                            normalized_url.lower().endswith(ext) for ext in [".pdf", ".xlsx", ".docx", ".csv", ".jpg", ".png", ".gif"]
                        ):
                            pages_to_visit.add(normalized_url)
        except Exception as e:
            print(f"Failed to process page for link discovery {current_url}: {e}")
            
    return list(all_links)

def html_to_markdown(element, base_url):
    """Recursively convert BeautifulSoup HTML elements into clean, relation-preserving Markdown."""
    if element is None:
        return ""
        
    # Check for comments
    if isinstance(element, Comment):
        return ""
        
    # If it is a string/text node, return its text
    if isinstance(element, str):
        return str(element)
        
    name = element.name
    if not name:
        return ""
        
    # Decompose unwanted boilerplate
    if name in ["script", "style", "header", "footer", "nav", "noscript"]:
        return ""
        
    # Check if this element matches the header or footer IDs/classes to avoid duplicating content
    element_id = element.get("id", "") or ""
    element_classes = element.get("class", []) or []
    
    if element_id in ["rs-header", "rs-footer"] or any(c in ["rs-header", "rs-footer", "full-width-header"] for c in element_classes):
        return ""
        
    # If the tag is an image
    if name == "img":
        alt = element.get("alt", "") or "Image"
        src = element.get("src", "")
        if src:
            abs_src = urljoin(base_url, src)
            return f"![{alt}]({abs_src})\n\n"
        return ""
        
    # If the tag is an anchor
    if name == "a":
        href = element.get("href", "")
        inner_content = "".join(html_to_markdown(c, base_url) for c in element.children).strip()
        if not inner_content:
            return ""
        if href:
            # Resolve relative URL to absolute URL
            abs_href = urljoin(base_url, href)
            return f"[{inner_content}]({abs_href})"
        return inner_content
        
    # Inline styles
    if name in ["strong", "b"]:
        inner_content = "".join(html_to_markdown(c, base_url) for c in element.children).strip()
        return f"**{inner_content}**" if inner_content else ""
        
    if name in ["em", "i"]:
        inner_content = "".join(html_to_markdown(c, base_url) for c in element.children).strip()
        return f"*{inner_content}*" if inner_content else ""
        
    if name == "br":
        return "\n"
        
    # Headings
    if name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(name[1])
        inner_content = "".join(html_to_markdown(c, base_url) for c in element.children).strip()
        return f"\n\n{'#' * level} {inner_content}\n\n" if inner_content else ""
        
    # Paragraphs / structural tags
    if name in ["p", "div", "section", "article", "main"]:
        inner_content = "".join(html_to_markdown(c, base_url) for c in element.children).strip()
        return f"\n\n{inner_content}\n\n" if inner_content else ""
        
    # Lists
    if name in ["ul", "ol"]:
        list_items = []
        is_ordered = (name == "ol")
        index = 1
        for child in element.children:
            if child.name == "li":
                item_content = "".join(html_to_markdown(c, base_url) for c in child.children).strip()
                if item_content:
                    prefix = f"{index}. " if is_ordered else "- "
                    list_items.append(f"{prefix}{item_content}")
                    index += 1
        return "\n\n" + "\n".join(list_items) + "\n\n"
        
    # GFM Tables
    if name == "table":
        markdown_table = []
        rows = element.find_all("tr")
        if not rows:
            return ""
            
        max_cols = 0
        table_data = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            cell_texts = []
            for cell in cells:
                cell_text = "".join(html_to_markdown(c, base_url) for c in cell.children).strip()
                # Replace multiple whitespace/newlines in cells with a single space
                cell_text = re.sub(r'\s+', ' ', cell_text)
                cell_texts.append(cell_text)
            if cell_texts:
                max_cols = max(max_cols, len(cell_texts))
                # Detect header rows by th element presence
                table_data.append((row.find("th") is not None, cell_texts))
                
        if not table_data:
            return ""
            
        def pad_row(row_cells, length):
            return row_cells + [""] * (length - len(row_cells))
            
        # Determine table header
        first_row_is_header = table_data[0][0]
        header_row = table_data[0][1]
        start_idx = 1
        
        padded_header = pad_row(header_row, max_cols)
        markdown_table.append("| " + " | ".join(padded_header) + " |")
        markdown_table.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        for is_th, cells in table_data[start_idx:]:
            padded_cells = pad_row(cells, max_cols)
            markdown_table.append("| " + " | ".join(padded_cells) + " |")
            
        return "\n\n" + "\n".join(markdown_table) + "\n\n"
        
    # Default fallback: traverse children
    return "".join(html_to_markdown(c, base_url) for c in element.children)

def clean_markdown(text):
    """Normalize whitespace and remove excessive newlines."""
    # Normalize multiple newlines to max 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove whitespace at start/end of lines
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_binary_file(response, url, output_dir):
    """Download binary files to a secure temporary path and parse their content."""
    content_type = response.headers.get("content-type", "").lower()
    ext_map = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif"
    }
    
    # Identify extension from URL or content type
    chosen_ext = None
    for mime_type, ext in ext_map.items():
        if mime_type in content_type or url.lower().endswith(ext):
            chosen_ext = ext
            break
            
    if not chosen_ext:
        return response.text
        
    # Use temporary file to prevent race conditions/name collisions
    with tempfile.NamedTemporaryFile(suffix=chosen_ext, dir=output_dir, delete=False) as temp_file:
        temp_path = temp_file.name
        try:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.close() # Close file handle so external parsers can read
            
            if chosen_ext == ".pdf":
                with pdfplumber.open(temp_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif chosen_ext == ".xlsx":
                wb = openpyxl.load_workbook(temp_path)
                text = ""
                for sheet in wb:
                    text += f"## Sheet: {sheet.title}\n"
                    for row in sheet.iter_rows(values_only=True):
                        text += "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
                    text += "\n"
            elif chosen_ext == ".docx":
                doc = docx.Document(temp_path)
                text = "\n".join(p.text for p in doc.paragraphs)
            else:  # Image
                text = pytesseract.image_to_string(Image.open(temp_path))
                
            return text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def download_and_convert(url, output_dir, playwright_context):
    """Download a URL (using Playwright for HTML pages, requests for files) and convert it to Markdown."""
    try:
        is_binary = any(url.lower().endswith(ext) for ext in [".pdf", ".xlsx", ".docx", ".csv", ".jpg", ".png", ".gif"])
        
        if is_binary:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            markdown_content = parse_binary_file(response, url, output_dir)
        else:
            # HTML page: fetch via Playwright context
            page = playwright_context.new_page()
            print(f"Loading page via Playwright: {url}")
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000) # Ensure JS dynamic lists render fully
            
            html_content = page.content()
            page.close()
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Decompose the header and footer in our DOM tree
            header = soup.find("header", id="rs-header")
            if header:
                header.decompose()
            footer = soup.find("footer", id="rs-footer")
            if footer:
                footer.decompose()
                
            raw_markdown = html_to_markdown(soup, url)
            markdown_content = clean_markdown(raw_markdown)
            
        if not markdown_content:
            print(f"⚠ Warning: Empty content for {url}")
            return None
            
        # Create a unique clean filename incorporating query parameters (essential for dynamic pages)
        parsed_url = urlparse(url)
        clean_path = parsed_url.path.strip('/')
        if parsed_url.query:
            clean_query = re.sub(r'[^\w\-_\.]', '_', parsed_url.query)
            clean_path = f"{clean_path}_{clean_query}"
            
        filename = re.sub(r'[^\w\-_\.]', '_', clean_path) or "index"
        if not filename.endswith('.md'):
            filename = f"{filename}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Source URL: {url}\n\n{markdown_content}")
            
        print(f"✓ Processed: {url}")
        return markdown_content
    except Exception as e:
        print(f"✗ Failed: {url} – {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with sync_playwright() as p:
        print("Phase 1: Discovering links using Playwright browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        urls = discover_links(BASE_URL, context)
        print(f"Found {len(urls)} URLs to process.\n")
        
        print("Phase 2: Downloading and converting using Playwright & Requests...")
        combined_md = f"# IIT Jammu Electrical Engineering Department – Crawled Content\n\nBase URL: {BASE_URL}\n\n---\n\n"
        
        for url in urls:
            md = download_and_convert(url, OUTPUT_DIR, context)
            if md:
                combined_md += f"\n\n---\n\n{md}"
                
        # Save combined file
        combined_path = os.path.join(OUTPUT_DIR, "00_combined_ee_site.md")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_md)
            
        browser.close()
        
    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder.")
