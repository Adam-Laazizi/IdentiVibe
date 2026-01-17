"""
LinkedIn scraping via Apify actors.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_urls(args):
    if not args:
        return []
    if len(args) == 1 and args[0].endswith(".txt"):
        path = Path(args[0])
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [a.strip() for a in args if a.strip()]


def _normalize_profile_input(value):
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if " " in value:
        return value
    return f"https://www.linkedin.com/in/{value}"


def _run_actor(client, actor_id, run_input):
    run = client.actor(actor_id).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])
    return list(dataset.iterate_items())


class LinkedInScraper:
    def __init__(self, api_key, profile_or_username):
        self.api_key = api_key
        self.profile_or_username = profile_or_username

    def git_payload(self):
        urls = [_normalize_profile_input(self.profile_or_username)]
        return _run_for_urls(urls, api_key=self.api_key)


def _run_for_urls(urls, api_key=None):
    if not urls:
        print("[FATAL] Provide at least one LinkedIn profile URL or a .txt file.")
        sys.exit(1)

    token = api_key or os.getenv("APIFY_TOKEN")
    if not token:
        print("[FATAL] Missing APIFY_TOKEN in environment or .env.")
        sys.exit(1)

    profile_actor = os.getenv(
        "LINKEDIN_PROFILE_ACTOR", "apimaestro/linkedin-profile-detail"
    )
    posts_actor = os.getenv(
        "LINKEDIN_POSTS_ACTOR", "apimaestro/linkedin-profile-posts"
    )

    client = ApifyClient(token)

    print(f"[*] Fetching profiles using {profile_actor}...")
    if profile_actor == "bebity/linkedin-premium-actor":
        query_type = "URL"
        queries = urls
        if urls and not urls[0].startswith("http"):
            query_type = "NAME"
        run_input = {
            "action": "Get Profiles",
            "queries": queries,
            "queryType": query_type,
            "limit": 1,
        }
    elif profile_actor == "dev_fusion/Linkedin-Profile-Scraper":
        run_input = {"profileUrls": urls}
    elif profile_actor == "apimaestro/linkedin-profile-detail":
        run_input = {"profileUrlOrUsername": urls}
    else:
        run_input = {"profileUrlOrUsername": urls}
    profile_items = _run_actor(client, profile_actor, run_input)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiles_file = OUTPUT_DIR / f"profiles_{timestamp}.json"
    profiles_file.write_text(json.dumps(profile_items, indent=2))
    print(f"[OK] Profiles saved to: {profiles_file}")

    posts_payloads = []
    if posts_actor:
        print(f"[*] Fetching posts using {posts_actor}...")
        for url in urls:
            try:
                if posts_actor == "supreme_coder/linkedin-post":
                    run_input = {"sourceUrls": [url], "limitPerSource": 10}
                elif posts_actor == "apimaestro/linkedin-posts-search-scraper-no-cookies":
                    run_input = {
                        "searchKeyword": f"\"{url}\"",
                        "sortType": "Relevance",
                        "pageNumber": 1,
                        "resultLimit": 50,
                    }
                elif posts_actor == "apimaestro/linkedin-profile-posts":
                    run_input = {
                        "profileUrlOrUsername": url,
                        "pageNumber": 1,
                        "resultLimit": 100,
                    }
                else:
                    run_input = {"profileUrls": [url]}
                posts = _run_actor(client, posts_actor, run_input)
                posts_payloads.append({"profileUrl": url, "items": posts})
            except Exception as exc:
                posts_payloads.append({"profileUrl": url, "error": str(exc)})
        posts_file = OUTPUT_DIR / f"posts_{timestamp}.json"
        posts_file.write_text(json.dumps(posts_payloads, indent=2))
        print(f"[OK] Posts saved to: {posts_file}")
    else:
        print("[!] LINKEDIN_POSTS_ACTOR not set; skipping posts/reactions/comments.")

    combined = []
    for i, url in enumerate(urls):
        profile = profile_items[i] if i < len(profile_items) else {}
        name = (
            profile.get("basic_info", {}).get("fullname")
            or profile.get("full_name")
            or profile.get("name")
            or ""
        )
        combined.append(
            {
                "profile_url": url,
                "name": name,
                "profile_data": profile,
                "posts": [],
                "followers_posts": [],
            }
        )
    if posts_payloads:
        for entry in combined:
            for post_group in posts_payloads:
                if post_group.get("profileUrl") == entry["profile_url"]:
                    entry["posts"] = post_group.get("items", [])

    combined_file = OUTPUT_DIR / f"combined_{timestamp}.json"
    combined_file.write_text(json.dumps(combined, indent=2))
    print(f"[OK] Combined JSON saved to: {combined_file}")
    print(json.dumps(combined, indent=2))
    return combined


def main():
    args = sys.argv[1:]
    urls = _read_urls(args)
    if urls:
        urls = [_normalize_profile_input(u) for u in urls]
    _run_for_urls(urls)


if __name__ == "__main__":
    main()
