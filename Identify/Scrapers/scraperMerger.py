"""
ScraperMerger - Simple orchestrator that calls all scrapers and merges their JSON outputs.

Usage:
    python scraperMerger.py --youtube "@mkbhd" --instagram "mkbhd" --reddit "technology" --linkedin "satyanadella"
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # overall.AI/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Output directory for JSON files
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scrape_youtube(handle: str) -> dict:
    """Scrape YouTube and return payload. Returns empty structure on failure."""
    if not handle:
        return {"platform": "youtube", "users": [], "error": "No handle provided"}

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"platform": "youtube", "users": [], "error": "YOUTUBE_API_KEY not set"}

    try:
        from Identify.Scrapers.Youtube.youTubeScraper import YouTubeScraper

        scraper = YouTubeScraper(api_key=api_key, target=handle, vids=5, comments=50)
        payload = scraper.get_payload()
        payload["platform"] = "youtube"
        return payload
    except Exception as e:
        return {"platform": "youtube", "users": [], "error": str(e)}


def scrape_instagram(handle: str) -> dict:
    """Scrape Instagram and return payload. Returns empty structure on failure."""
    if not handle:
        return {"platform": "instagram", "users": [], "error": "No handle provided"}

    token = os.getenv("APIFY_TOKEN")
    if not token:
        return {"platform": "instagram", "users": [], "error": "APIFY_TOKEN not set"}

    try:
        from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper

        scraper = InstagramScraper(
            apify_token=token,
            target=handle,
            settings={
                "posts": 5,              # Seed account posts to scrape
                "comments": 5,          # Total comments to collect
                "sample": 5,            # Number of users to sample
                "user_posts": 5,         # Posts per user for captions
                "max_comments_per_user": 5,
                "cache_dir": "./cache",
            }
        )
        payload = scraper.get_payload()
        payload["platform"] = "instagram"
        return payload
    except Exception as e:
        return {"platform": "instagram", "users": [], "error": str(e)}


def scrape_reddit(subreddit: str) -> dict:
    """Scrape Reddit and return payload. Returns empty structure on failure."""
    if not subreddit:
        return {"platform": "reddit", "users": [], "error": "No subreddit provided"}

    try:
        from Identify.Scrapers.Reddit.RedditScraper import RedditScraper

        scraper = RedditScraper()
        payload = scraper.get_payload(target_subreddit=subreddit)
        return payload  # Already has "platform": "reddit"
    except Exception as e:
        return {"platform": "reddit", "users": [], "error": str(e)}


def scrape_linkedin(profile: str) -> dict:
    """Scrape LinkedIn and return payload. Returns empty structure on failure."""
    if not profile:
        return {"platform": "linkedin", "users": [], "error": "No profile provided"}

    token = os.getenv("APIFY_TOKEN")
    if not token:
        return {"platform": "linkedin", "users": [], "error": "APIFY_TOKEN not set"}

    try:
        from Identify.Scrapers.linkedin.scrape_linkedin import LinkedInScraper

        scraper = LinkedInScraper(api_key=token, profile_or_username=profile)
        payload = scraper.get_payload()
        return payload  # Already has "platform": "linkedin"
    except Exception as e:
        return {"platform": "linkedin", "users": [], "error": str(e)}


def save_json(data: dict, filename: str) -> Path:
    """Save data to JSON file and return the path."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath


def merge_all(
    youtube: Optional[str] = None,
    instagram: Optional[str] = None,
    reddit: Optional[str] = None,
    linkedin: Optional[str] = None,
) -> dict:
    """
    Run all scrapers, save individual JSON files, and merge into one.

    Returns the merged payload.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # Scrape each platform and save individual JSON files
    print(f"[*] Scraping YouTube: {youtube or '(skipped)'}")
    yt_data = scrape_youtube(youtube) if youtube else {"platform": "youtube", "users": []}
    save_json(yt_data, f"youtube_{timestamp}.json")
    results["youtube"] = yt_data
    print(f"    -> {len(yt_data.get('users', []))} users")

    print(f"[*] Scraping Instagram: {instagram or '(skipped)'}")
    ig_data = scrape_instagram(instagram) if instagram else {"platform": "instagram", "users": []}
    save_json(ig_data, f"instagram_{timestamp}.json")
    results["instagram"] = ig_data
    print(f"    -> {len(ig_data.get('users', []))} users")

    print(f"[*] Scraping Reddit: r/{reddit or '(skipped)'}")
    rd_data = scrape_reddit(reddit) if reddit else {"platform": "reddit", "users": []}
    save_json(rd_data, f"reddit_{timestamp}.json")
    results["reddit"] = rd_data
    print(f"    -> {len(rd_data.get('users', []))} users")

    print(f"[*] Scraping LinkedIn: {linkedin or '(skipped)'}")
    li_data = scrape_linkedin(linkedin) if linkedin else {"platform": "linkedin", "users": []}
    save_json(li_data, f"linkedin_{timestamp}.json")
    results["linkedin"] = li_data
    print(f"    -> {len(li_data.get('users', []))} users")

    # Merge all into one payload - array of results
    total_users = sum(len(p.get("users", [])) for p in results.values())

    # Build array of scraper outputs (only include platforms that were scraped)
    merged_array = []
    for platform_data in results.values():
        # Only include if platform was actually scraped (has users or no error)
        if platform_data.get("users") or "error" not in platform_data:
            merged_array.append(platform_data)

    # Save merged output as JSON array
    merged_path = save_json(merged_array, f"merged_{timestamp}.json")
    print(f"\n[OK] Merged output saved to: {merged_path}")
    print(f"[OK] Total users across all platforms: {total_users}")

    return merged_array


if __name__ == "__main__":
    # Parse CLI arguments
    args = sys.argv[1:]

    youtube_handle = None
    instagram_handle = None
    reddit_sub = None
    linkedin_profile = None

    i = 0
    while i < len(args):
        if args[i] == "--youtube" and i + 1 < len(args):
            youtube_handle = args[i + 1]
            i += 2
        elif args[i] == "--instagram" and i + 1 < len(args):
            instagram_handle = args[i + 1]
            i += 2
        elif args[i] == "--reddit" and i + 1 < len(args):
            reddit_sub = args[i + 1]
            i += 2
        elif args[i] == "--linkedin" and i + 1 < len(args):
            linkedin_profile = args[i + 1]
            i += 2
        else:
            i += 1

    if not any([youtube_handle, instagram_handle, reddit_sub, linkedin_profile]):
        print("Usage: python scraperMerger.py [--youtube HANDLE] [--instagram HANDLE] [--reddit SUBREDDIT] [--linkedin PROFILE]")
        print("\nExample:")
        print('  python scraperMerger.py --youtube "@mkbhd" --reddit "technology"')
        sys.exit(1)

    merge_all(
        youtube=youtube_handle,
        instagram=instagram_handle,
        reddit=reddit_sub,
        linkedin=linkedin_profile,
    )
