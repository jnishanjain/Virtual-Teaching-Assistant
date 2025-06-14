#!/usr/bin/env python3
"""
batch_md_to_json.py

1. GET  https://tds.s-anand.net/_sidebar.md
2. Parse all Markdown links (to .md files)
3. For each .md:
     - GET raw Markdown
     - Convert to HTML (markdown lib)
     - Parse HTML with BeautifulSoup for text
4. Save a JSON list of:
   { "page": "<slug>", "markdown": "...", "html": "...", "text": "..." }
   under data/course_content.json
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
import markdown as mdlib

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE       = "https://tds.s-anand.net"
SIDEBARURL = BASE + "/_sidebar.md"
OUTPUT     = "data/course_content.json"
HEADERS    = {"Accept": "text/markdown, */*", "User-Agent": "Mozilla/5.0"}
# ────────────────────────────────────────────────────────────────────────────────

def download_text(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text

def parse_sidebar(sidebar_md):
    """
    Extract all .md targets from the sidebar.
    e.g. lines like [Title](development-tools.md) or (tools/foo.md)
    """
    links = re.findall(r'\[.*?\]\(([^)]+\.md)\)', sidebar_md)
    # filter out _sidebar.md itself
    return sorted(set([l for l in links if not l.startswith("_")]))

def convert_and_parse(md_text):
    """Convert markdown → HTML, then parse via BS4."""
    html = mdlib.markdown(md_text, extensions=["fenced_code","tables"])
    soup = BeautifulSoup(html, "html.parser")
    # Extract plain text (you can tweak this as needed)
    text = soup.get_text("\n", strip=True)
    return html, text

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print("1) Downloading sidebar:", SIDEBARURL)
    sidebar_md = download_text(SIDEBARURL)

    md_files = parse_sidebar(sidebar_md)
    print(f"2) Found {len(md_files)} Markdown files:")
    for fn in md_files:
        print("   ", fn)
    
    all_pages = []
    for fn in md_files:
        url = BASE + "/" + fn
        print(f"\n→ Fetching: {url}")
        try:
            md = download_text(url)
        except Exception as e:
            print("   ✖ failed:", e)
            continue

        html, text = convert_and_parse(md)
        slug = fn.rsplit("/",1)[-1].rsplit(".md",1)[0]
        record = {
            "page":       slug,
            "markdown":   md,
            "html":       html,
            "text":       text
        }
        all_pages.append(record)

        # polite crawl
        time.sleep(0.2)

    # write out everything
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(all_pages)} pages to {OUTPUT}")

if __name__ == "__main__":
    main()
