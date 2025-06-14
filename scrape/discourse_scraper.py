import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
from tqdm import tqdm

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/"
CATEGORY_PATH = "/c/courses/tds-kb/34"
COOKIE_NAME = "_t"  # this may be _forum_session or similar depending on your browser

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)

# 🔐 Replace this with your actual session cookie string
SESSION_COOKIE_VALUE = '''
    bqNjR9p1Je8RAFrFTakscFwGk77ik%2BE9jp1cr0EYJXuLoTCyhfTJ04%2BkZrDROqqMQ5D0kGQ9amiCWvVDhcLXlYzA0LMDvukn6nWaDNWJHEml8RypXnV9Z9On5%2B4sW%2FkeDPFvSPnpVMhe%2Fau44vZ3r1oGGeDnaYi9oncz8IKXerm8ESEPLx2yl9DOUF1CmPEgfkFiMzLmi%2B9dLKQ1i1NW%2Fvs0itrlNwUdO8%2FrH7AK9rISoecbqIacv9xEOyFbHwTi3dm9orY3rCTtox8Zbq5In3lUElSan%2BefkE7aHMZG%2BWNnBFDGNiYsOnLyYTlposEr--Ysc9Gq0qAl2ArBg8--LwX3jsdPhSvSnTtPoUZTTA%3D%3D
    '''


def is_in_range(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return START_DATE <= dt <= END_DATE
    except Exception:
        return False


def get_topic_links(session):
    topic_links = []
    for page in tqdm(range(0, 10), desc="Fetching topic pages"):
        url = f"{BASE_URL}{CATEGORY_PATH}/l/latest.json?page={page}"
        res = session.get(url)
        if res.status_code != 200:
            print(f"Failed to load page {page}: {res.status_code}")
            break
        data = res.json()
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        for topic in topics:
            if is_in_range(topic["created_at"]):
                topic_links.append({
                    "id": topic["id"],
                    "slug": topic["slug"],
                    "title": topic["title"],
                    "created_at": topic["created_at"]
                })
    return topic_links


def get_post_content(session, topic_id, slug):
    url = f"{BASE_URL}/t/{slug}/{topic_id}.json"
    res = session.get(url)
    if res.status_code != 200:
        print(f"Failed to load topic {slug}: {res.status_code}")
        return ""
    data = res.json()
    posts = data.get("post_stream", {}).get("posts", [])
    content = "\n\n".join(p.get("cooked", "") for p in posts)
    return BeautifulSoup(content, "html.parser").get_text()


def scrape():
    session = requests.Session()
    session.cookies.set(COOKIE_NAME, SESSION_COOKIE_VALUE, domain="discourse.onlinedegree.iitm.ac.in")

    os.makedirs("data", exist_ok=True)

    topics = get_topic_links(session)
    all_data = []
    for topic in tqdm(topics, desc="Fetching topic content"):
        content = get_post_content(session, topic["id"], topic["slug"])
        all_data.append({
            "url": f"{BASE_URL}/t/{topic['slug']}/{topic['id']}",
            "title": topic["title"],
            "created_at": topic["created_at"],
            "text": content
        })

    with open("data/discourse_posts.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"✅ Scraped {len(all_data)} topics into data/discourse_posts.json")


if __name__ == "__main__":
    scrape()
