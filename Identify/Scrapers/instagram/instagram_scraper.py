"""
Instagram Scraper - Main entry point.

Usage:
    from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper

    scraper = InstagramScraper()
    payload = scraper.get_payload("uoft")
    # Returns: {"seed_handle": "uoft", "users": [...]}
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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

    def __init__(self, payload_path: str = "payload.json"):
        self.payload_path = payload_path

    def _load_payload(self) -> Dict[str, Any]:
        """Load configuration from payload file."""
        path = Path(self.payload_path)
        if not path.exists():
            raise FileNotFoundError(f"Payload file not found: {self.payload_path}")

        with open(path) as f:
            return json.load(f)

    def get_payload(self, target: str) -> dict:
        """
        Scrape Instagram data for commenters on a seed account's posts.

        Args:
            target: Instagram handle of the seed account (e.g., "uoft")

        Returns:
            Dictionary with structure: {"seed_handle": "...", "users": [...]}
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

        logger.info(f"Scraping Instagram for seed: @{target}")
        logger.info(f"Config: posts={posts_limit}, comments={comments_limit}, sample={sample_size}")

        # Initialize client
        client = ApifyClient(token=apify_token, cache_dir=cache_dir)

        # Step 1: Scrape seed account posts
        logger.info(f"Step 1: Scraping posts from @{target}...")
        seed_posts = client.scrape_profile_posts(
            username=target,
            results_limit=posts_limit,
        )

        if not seed_posts:
            raise ApifyError(f"No posts found for seed account @{target}")

        # Extract post URLs
        post_urls = extract_post_urls(seed_posts)
        logger.info(f"Found {len(post_urls)} posts from @{target}")

        if not post_urls:
            raise ApifyError("No post URLs extracted from seed account")

        # Step 2: Scrape comments from seed posts
        logger.info(f"Step 2: Scraping comments from {len(post_urls)} posts...")
        all_comments = client.scrape_post_comments(
            post_urls=post_urls,
            results_limit=comments_limit,
        )
        logger.info(f"Scraped {len(all_comments)} total comments")

        if not all_comments:
            raise ApifyError("No comments found on seed posts")

        # Step 3: Bundle comments by user
        logger.info("Step 3: Bundling comments by username...")
        user_comments = bundle_seed_comments(
            comments=all_comments,
            max_comments_per_user=max_comments_per_user,
            deduplicate=True,
        )
        logger.info(f"Found {len(user_comments)} unique commenters")

        if not user_comments:
            raise ApifyError("No valid comments found after bundling")

        # Step 4: Sample usernames
        logger.info(f"Step 4: Sampling {sample_size} usernames...")
        sampled_users = sample_usernames(
            user_comments=user_comments,
            sample_size=sample_size,
        )
        logger.info(f"Sampled {len(sampled_users)} users")

        # Step 5: Enrich with captions
        logger.info(f"Step 5: Enriching {len(sampled_users)} users with captions...")
        bundles, skipped = enrich_with_captions(
            client=client,
            sampled_usernames=sampled_users,
            user_comments=user_comments,
            user_posts_limit=user_posts_limit,
        )

        # Format output to match YouTubeScraper pattern
        users = [
            {
                "username": b["username"],
                "comments": b["comments"],
                "captions": b["captions"],
            }
            for b in bundles
        ]

        logger.info(f"Created {len(users)} user bundles, skipped {skipped} users")

        return {"seed_handle": target, "users": users}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python instagram_scraper.py <seed_handle> [output_path]")
        sys.exit(1)

    handle = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "instagram_bundles.json"

    try:
        scraper = InstagramScraper()
        result = scraper.get_payload(handle)

        # Write output
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Success: {len(result['users'])} user bundles created")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
