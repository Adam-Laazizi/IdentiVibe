"""
Instagram Scraper - Main entry point.

Usage:
    from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper

    scraper = InstagramScraper(target="uoft")
    payload = scraper.get_payload()
    # Returns: {"seed_handle": "uoft", "users": [...]}
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import sys
import os

# 1. Get the path of the 'Scrapers' folder (which is the parent of the current file)
# This handles the case where you are in root/Identify/Scrapers/Youtube/
scrapers_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Add 'Scrapers' to the system path so Python can find SocialScraper.py
if scrapers_dir not in sys.path:
    sys.path.insert(0, scrapers_dir)

# 3. Import with the CORRECT CASE (Capital S)
from socialScraper import SocialScraper

from connectors.instagram.apify_client import ApifyClient, ApifyError
from connectors.instagram.bundler import (
    bundle_seed_comments,
    enrich_with_captions,
    extract_post_urls,
    sample_usernames,
)

logger = logging.getLogger(__name__)

# InstagramScraper is now a subclass of SocialScraper and sister to YouTubeScraper
class InstagramScraper(SocialScraper):
    """Handles all interactions with Instagram via Apify."""

    def __init__(self, target: str, payload_path: str = "payload.json"):
        """
        Standardized constructor matching sister classes.
        :param target: Instagram handle of the seed account (e.g., "uoft")
        :param payload_path: Path to configuration settings
        """
        self.target = target
        self.payload_path = payload_path

    def _load_payload(self) -> Dict[str, Any]:
        """Load configuration from payload file."""
        path = Path(self.payload_path)
        if not path.exists():
            raise FileNotFoundError(f"Payload file not found: {self.payload_path}")

        with open(path) as f:
            return json.load(f)

    def get_payload(self) -> dict:
        """
        Implementation of the SocialScraper interface.
        Scrapes Instagram data for commenters on the seed account's posts.
        :return: Dictionary with structure: {"seed_handle": "...", "users": [...]}
        """
        # Load config
        config = self._load_payload()

        apify_token = config.get("apify_token")
        posts_limit = config.get("posts", 10)
        comments_limit = config.get("comments", 150)
        sample_size = config.get("sample", 250)
        user_posts_limit = config.get("user_posts", 10)
        max_comments_per_user = config.get("max_comments_per_user", 50)
        cache_dir = config.get("cache_dir", "./cache")

        logger.info(f"Scraping Instagram for seed: @{self.target}")

        # Initialize client
        client = ApifyClient(token=apify_token, cache_dir=cache_dir)

        # Step 1: Scrape seed account posts
        logger.info(f"Step 1: Scraping posts from @{self.target}...")
        seed_posts = client.scrape_profile_posts(
            username=self.target,
            results_limit=posts_limit,
        )

        if not seed_posts:
            raise ApifyError(f"No posts found for seed account @{self.target}")

        post_urls = extract_post_urls(seed_posts)

        # Step 2: Scrape comments from seed posts
        all_comments = client.scrape_post_comments(
            post_urls=post_urls,
            results_limit=comments_limit,
        )

        # Step 3: Bundle comments by user
        user_comments = bundle_seed_comments(
            comments=all_comments,
            max_comments_per_user=max_comments_per_user,
            deduplicate=True,
        )

        # Step 4: Sample usernames
        sampled_users = sample_usernames(
            user_comments=user_comments,
            sample_size=sample_size,
        )

        # Step 5: Enrich with captions
        bundles, skipped = enrich_with_captions(
            client=client,
            sampled_usernames=sampled_users,
            user_comments=user_comments,
            user_posts_limit=user_posts_limit,
        )

        # Format output to match the standardized 'Identity Payload'
        users = [
            {
                "username": b["username"],
                "comments": b["comments"],
                "captions": b["captions"],
            }
            for b in bundles
        ]

        return {"seed_handle": self.target, "users": users}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python instagram_scraper.py <seed_handle>")
        sys.exit(1)

    handle = sys.argv[1]

    try:
        # Initializing with target handle
        scraper = InstagramScraper(target=handle)
        result = scraper.get_payload()

        with open("instagram_bundles.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Success: {len(result['users'])} user bundles created")
    except Exception as e:
        print(f"Error: {e}")