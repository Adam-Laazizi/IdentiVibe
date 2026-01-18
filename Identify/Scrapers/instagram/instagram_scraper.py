"""
Instagram Scraper - Main entry point.

Usage (Programmatic):
    scraper = InstagramScraper(apify_token="TOKEN", settings={"posts": 5, "sample": 10})
    payload = scraper.get_payload("uoft")
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

try:
    from Identify.Scrapers.SocialScraper import SocialScraper
except ImportError:
    # Allow running standalone from the instagram directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SocialScraper import SocialScraper

from connectors.instagram.apify_client import ApifyClient, ApifyError
from connectors.instagram.bundler import (
    bundle_seed_comments,
    enrich_with_captions,
    extract_post_urls,
    sample_usernames,
)

logger = logging.getLogger(__name__)

class InstagramScraper(SocialScraper):
    """Handles all interactions with Instagram via Apify."""

    def __init__(self, apify_token: str, settings: Dict[str, Any] = None):
        if not apify_token:
            raise ValueError("An Apify API token must be provided to initialize the scraper.")
            
        self.apify_token = apify_token
        
        # Default settings are used if no settings dict is provided.
        # These are easily overridden during website integration.
        self.settings = settings or {
            "posts": 10,
            "comments": 150,
            "sample": 250,
            "user_posts": 10,
            "max_comments_per_user": 50,
            "cache_dir": "./cache"
        }

    @staticmethod
    def _normalize_handle(target: str) -> str:
        """Normalize and validate an Instagram handle or profile URL."""
        if not isinstance(target, str):
            raise ValueError("Seed handle must be a string.")

        handle = target.strip()
        if not handle:
            return ""

        # Handle full URL inputs
        if handle.startswith("http://") or handle.startswith("https://"):
            parsed = urlparse(handle)
            if parsed.netloc not in {"instagram.com", "www.instagram.com"}:
                raise ValueError("Handle must be a valid Instagram username or profile URL.")

            path = parsed.path.strip("/")
            if not path:
                raise ValueError("Instagram profile URL missing username.")
            handle = path.split("/", 1)[0]

        # Clean leading @ and whitespace
        handle = handle.lstrip("@").strip()
        return handle

    def get_payload(self, target: str) -> dict:
        """
        Scrape Instagram data based on the initialized settings.

        Args:
            target: The Instagram handle to scrape.

        Returns:
            A dictionary containing the seed handle and the bundled user data.
        """
        target = self._normalize_handle(target)
        if not target:
            return {"seed_handle": "", "users": []}

        # Extract operational limits from settings
        posts_limit = self.settings.get("posts", 10)
        comments_limit = self.settings.get("comments", 150)
        sample_size = self.settings.get("sample", 250)
        user_posts_limit = self.settings.get("user_posts", 10)
        max_comments_per_user = self.settings.get("max_comments_per_user", 50)
        cache_dir = self.settings.get("cache_dir", "./cache")

        logger.info(f"Starting scrape for @{target} with sample_size={sample_size}")

        # Initialize the Apify communication client
        client = ApifyClient(token=self.apify_token, cache_dir=cache_dir)

        # Step 1: Scrape seed account posts
        logger.info(f"Step 1: Scraping {posts_limit} posts from @{target}...")
        seed_posts = client.scrape_profile_posts(
            username=target,
            results_limit=posts_limit,
        )

        if not seed_posts:
            logger.warning(f"No posts found for handle: @{target}")
            return {"seed_handle": target, "users": []}

        post_urls = extract_post_urls(seed_posts)
        if not post_urls:
            raise ApifyError("No post URLs could be extracted from the seed account posts.")

        # Step 2: Scrape comments from those posts
        logger.info(f"Step 2: Scraping {comments_limit} total comments...")
        all_comments = client.scrape_post_comments(
            post_urls=post_urls,
            results_limit=comments_limit,
        )

        if not all_comments:
            raise ApifyError("No comments were found on the targeted posts.")

        # Step 3: Bundle comments by user
        logger.info("Step 3: Bundling and deduplicating user comments...")
        user_comments = bundle_seed_comments(
            comments=all_comments,
            max_comments_per_user=max_comments_per_user,
            deduplicate=True,
        )

        # Step 4: Sample the commenters based on the provided sample size
        logger.info(f"Step 4: Sampling up to {sample_size} unique users...")
        sampled_users = sample_usernames(
            user_comments=user_comments,
            sample_size=sample_size,
        )

        # Step 5: Enrich the sampled users with their own captions/post history
        logger.info(f"Step 5: Enriching {len(sampled_users)} users with personal captions...")
        bundles, skipped = enrich_with_captions(
            client=client,
            sampled_usernames=sampled_users,
            user_comments=user_comments,
            user_posts_limit=user_posts_limit,
        )

        # Final formatting for consumption by GeminiEnricher
        users = [
            {
                "username": b["username"],
                "comments": b["comments"],
                "captions": b["captions"],
            }
            for b in bundles
        ]

        logger.info(f"Scrape complete: {len(users)} user bundles created ({skipped} users skipped).")
        return {"seed_handle": target, "users": users}

# Support for local execution/debugging
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python instagram_scraper.py <handle>")
        sys.exit(1)

    # For local tests, we still look for a local token file,
    # but the class itself no longer depends on it.
    try:
        with open("payload.json") as f:
            local_config = json.load(f)
        
        token = local_config.get("apify_token")
        target_handle = sys.argv[1]
        
        # Instantiate with the new pattern
        scraper = InstagramScraper(apify_token=token, settings=local_config)
        result = scraper.get_payload(target_handle)
        
        # Write results to a file for inspection
        with open("instagram_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Success! Generated {len(result['users'])} user profiles.")
    except Exception as e:
        print(f"Error during execution: {e}")