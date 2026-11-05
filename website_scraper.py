"""
Website Scraper Module - Extracts text content from college website.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import random
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader
from io import BytesIO

load_dotenv()

COLLEGE_URL = os.getenv("COLLEGE_URL", "https://www.srmvalliammai.ac.in/")
DEPARTMENT_URL = os.getenv(
    "DEPARTMENT_URL",
    urljoin(COLLEGE_URL, "/department/department-of-artificial-intelligence-and-data-science"),
)
MAX_PAGES = int(os.getenv("WEBSITE_MAX_PAGES", "150"))
SUPPORTED_URLS = {
    "home": COLLEGE_URL,
    "events": urljoin(COLLEGE_URL, "/events"),
    "placements": urljoin(COLLEGE_URL, "/placements"),
    "news": urljoin(COLLEGE_URL, "/news"),
    "department": urljoin(COLLEGE_URL, "/department"),
    "artificial_intelligence_and_data_science": DEPARTMENT_URL,
    "about": urljoin(COLLEGE_URL, "/about"),
    "admission": urljoin(COLLEGE_URL, "/admission"),
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
}

MIN_DELAY = 1.0
MAX_DELAY = 3.0


def normalize_url(url: str) -> str:
    """Remove fragments and normalize a URL for crawl de-duplication."""
    without_fragment, _ = urldefrag(url)
    parsed = urlparse(without_fragment)
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path).geturl()


def is_valid_url(url: str, base_url: str) -> bool:
    """Check if URL belongs to the same domain and uses HTTP/HTTPS."""
    try:
        parsed = urlparse(url)
        base_parsed = urlparse(base_url)
        
        # Same domain
        if parsed.netloc.lower() != base_parsed.netloc.lower():
            return False
        
        # Use HTTP/HTTPS
        if not parsed.scheme in ("http", "https"):
            return False
        
        # Skip fragments, common exclude paths
        exclude_paths = ("/login", "/admin", "/private", "/dashboard")
        if any(parsed.path.lower().startswith(path) for path in exclude_paths):
            return False
        
        return True
    except Exception:
        return False


def extract_visible_text(soup: BeautifulSoup) -> str:
    """
    Extract visible text from BeautifulSoup object.
    Remove navigation, sidebar, footer, header, ads.
    """
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.decompose()
    
    # Get text from common content tags
    content_tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span"])
    
    text_parts = []
    for element in content_tags:
        text = element.get_text(strip=True)
        if len(text) > 30:  # Ignore very short texts
            text_parts.append(text)
    
    # Also try getting text from article if present
    article = soup.find("article")
    if article:
        article_text = article.get_text(strip=True)
        if len(article_text) > 50:
            text_parts.append(article_text)
    
    return "\n".join(text_parts)


def scrape_page(url: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """Scrape a single URL and extract text content."""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        is_pdf = "application/pdf" in content_type or urlparse(url).path.lower().endswith(".pdf")
        if is_pdf:
            reader = PdfReader(BytesIO(response.content))
            content = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            title = Path(urlparse(url).path).name or "PDF document"
            return {
                "url": url,
                "title": title,
                "content": content,
                "success": bool(content),
                "links": [],
            }

        soup = BeautifulSoup(response.content, "lxml")
        discovered_links = [
            normalize_url(urljoin(url, link["href"]))
            for link in soup.find_all("a", href=True)
        ]
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
        
        # Try to extract better title from h1/h2
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
        
        # Extract main content
        content = extract_visible_text(soup)
        
        if content:
            return {
                "url": url,
                "title": title,
                "content": content,
                "success": True,
                "links": discovered_links,
            }
        else:
            print(f"   WARNING: No content extracted from: {url}")
            return {
                "url": url,
                "title": title,
                "content": "",
                "success": False,
                "links": [],
            }
            
    except Exception as e:
        print(f"   ERROR: Error scraping {url}: {e}")
        return {
            "url": url,
            "title": "",
            "content": "",
            "success": False,
            "links": [],
        }


def crawl_website(base_url: str) -> List[Dict[str, Any]]:
    """
    Crawl the college website and extract content from key pages.
    Returns list of dictionaries with url, title, and content.
    """
    print(f"Crawling website: {base_url}")
    
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    
    all_pages = []
    queued_urls = [normalize_url(url) for url in SUPPORTED_URLS.values()]
    queued_urls.insert(0, normalize_url(DEPARTMENT_URL))
    visited_urls = set()

    while queued_urls and len(visited_urls) < MAX_PAGES:
        url = queued_urls.pop(0)
        if url in visited_urls or not is_valid_url(url, base_url):
            continue

        visited_urls.add(url)
        print(f"   Scraping ({len(visited_urls)}/{MAX_PAGES}): {url}")
        page_data = scrape_page(url, session)
        if page_data and page_data.get("success"):
            all_pages.append(page_data)

        for discovered_url in page_data.get("links", []) if page_data else []:
            if (
                discovered_url not in visited_urls
                and discovered_url not in queued_urls
                and is_valid_url(discovered_url, base_url)
            ):
                queued_urls.append(discovered_url)

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    
    successful = sum(1 for p in all_pages if p.get("success"))
    print(
        f"Crawling complete: {len(visited_urls)} URLs visited, "
        f"{successful} successfully scraped"
    )
    
    return all_pages


def scrape_all() -> List[Dict[str, Any]]:
    """Main entry point for website scraping."""
    all_pages = crawl_website(COLLEGE_URL)
    
    # Format for ingestion
    formatted_documents = []
    for page in all_pages:
        if page.get("content"):
            formatted_documents.append({
                "content": page["content"],
                "metadata": {
                    "source": "website",
                    "url": page["url"],
                    "title": page["title"],
                    "scraped_at": ""
                }
            })
    
    return formatted_documents