"""
Instagram Scraper - Main entry point.

Usage:
    from instagram_scraper import scrape_instagram

    bundles = scrape_instagram("uoft")
    # Returns: [{"username": "...", "comments": [...], "captions": [...]}]
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from connectors.instagram.apify_client import ApifyClient, ApifyError
from connectors.instagram.bundler import (
    bundle_seed_comments,
    enrich_with_captions,
    extract_post_urls,
    sample_usernames,
)

logger = logging.getLogger(__name__)


def load_payload(payload_path: str = "payload.json") -> Dict[str, Any]:
    """Load configuration from payload file."""
    path = Path(payload_path)
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")

    with open(path) as f:
        return json.load(f)


def scrape_instagram(
    seed_handle: str,
    payload_path: str = "payload.json",
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape Instagram data for commenters on a seed account's posts.

    Args:
        seed_handle: Instagram handle of the seed account (e.g., "uoft")
        payload_path: Path to JSON config file with scraping parameters
        output_path: Optional path to write output JSON. If None, only returns data.

    Returns:
        List of bundles: [{"username": "...", "comments": [...], "captions": [...]}]
    """
    # Load config
    config = load_payload(payload_path)

    posts_limit = config.get("posts", 10)
    comments_limit = config.get("comments", 150)
    sample_size = config.get("sample", 250)
    user_posts_limit = config.get("user_posts", 10)
    max_comments_per_user = config.get("max_comments_per_user", 50)
    cache_dir = config.get("cache_dir", "./cache")

    logger.info(f"Scraping Instagram for seed: @{seed_handle}")
    logger.info(f"Config: posts={posts_limit}, comments={comments_limit}, sample={sample_size}")

    # Initialize client
    client = ApifyClient(cache_dir=cache_dir)

    # Step 1: Scrape seed account posts
    logger.info(f"Step 1: Scraping posts from @{seed_handle}...")
    seed_posts = client.scrape_profile_posts(
        username=seed_handle,
        results_limit=posts_limit,
    )

    if not seed_posts:
        raise ApifyError(f"No posts found for seed account @{seed_handle}")

    # Extract post URLs
    post_urls = extract_post_urls(seed_posts)
    logger.info(f"Found {len(post_urls)} posts from @{seed_handle}")

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

    # Format output
    output_bundles = [
        {
            "username": b["username"],
            "comments": b["comments"],
            "captions": b["captions"],
        }
        for b in bundles
    ]

    logger.info(f"Created {len(output_bundles)} bundles, skipped {skipped} users")

    # Write output if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_bundles, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote output to {output_path}")

    return output_bundles


if __name__ == "__main__":
    import sys

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
        result = scrape_instagram(handle, output_path=output)
        print(f"Success: {len(result)} bundles created")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
