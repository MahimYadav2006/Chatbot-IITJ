from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def debug_url(url):
    print(f"\n========================================\nLoading: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
        
    soup = BeautifulSoup(html, "html.parser")
    print("Title:", soup.title.string if soup.title else "No Title")
    print("Text length:", len(soup.get_text()))
    print("Preview of text:")
    # Print non-whitespace lines
    lines = [l.strip() for l in soup.get_text().split("\n") if l.strip()]
    for line in lines[:15]:
        print("  >", line)

def debug():
    debug_url("https://iitjammu.ac.in/computer_science_engineering/faculty-list")
    debug_url("https://iitjammu.ac.in/computer_science_engineering/labs")

if __name__ == "__main__":
    debug()
