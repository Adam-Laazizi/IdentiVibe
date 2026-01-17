#!/usr/bin/env python3
"""
Instagram Bundle CLI - Scrape and bundle Instagram data using Apify.

Usage:
    python scripts/run_instagram_bundle.py \
        --seed uoft \
        --posts 10 \
        --comments 150 \
        --sample 250 \
        --user_posts 10 \
        --out instagram_bundles.json

Output format:
    [
      {
        "username": "john_doe",
        "comments": ["comment1", "comment2"],
        "captions": ["caption1", "caption2"]
      }
    ]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.instagram.apify_client import ApifyClient, ApifyError
from connectors.instagram.bundler import (
    bundle_seed_comments,
    enrich_with_captions,
    extract_post_urls,
    sample_usernames,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape Instagram comments and captions, bundle by user.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--seed",
        type=str,
        default="uoft",
        help="Seed Instagram account handle to scrape (default: uoft)",
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=10,
        help="Number of posts to scrape from seed account (default: 10)",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=150,
        help="Number of comments to scrape per post (default: 150)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=250,
        help="Number of unique commenters to sample (default: 250)",
    )
    parser.add_argument(
        "--user_posts",
        type=int,
        default=10,
        help="Number of posts to scrape per sampled user (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="instagram_bundles.json",
        help="Output JSON file path (default: instagram_bundles.json)",
    )
    parser.add_argument(
        "--max_comments_per_user",
        type=int,
        default=50,
        help="Maximum comments to keep per user (default: 50)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for caching Apify results (default: ./cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of Apify results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Instagram Bundle Scraper")
    logger.info("=" * 60)
    logger.info(f"Seed account: @{args.seed}")
    logger.info(f"Posts to scrape: {args.posts}")
    logger.info(f"Comments per post: {args.comments}")
    logger.info(f"Users to sample: {args.sample}")
    logger.info(f"Posts per user: {args.user_posts}")
    logger.info(f"Output file: {args.out}")
    logger.info("=" * 60)

    try:
        # Initialize Apify client
        client = ApifyClient(cache_dir=args.cache_dir)
        use_cache = not args.no_cache

        # Step 1: Scrape seed account posts
        logger.info(f"Step 1: Scraping posts from @{args.seed}...")
        seed_posts = client.scrape_profile_posts(
            username=args.seed,
            results_limit=args.posts,
        )

        if not seed_posts:
            logger.error(f"No posts found for seed account @{args.seed}")
            return 1

        # Extract post URLs
        post_urls = extract_post_urls(seed_posts)
        logger.info(f"Found {len(post_urls)} posts from @{args.seed}")

        if not post_urls:
            logger.error("No post URLs extracted from seed account")
            return 1

        # Step 2: Scrape comments from seed posts
        logger.info(f"Step 2: Scraping comments from {len(post_urls)} posts...")
        all_comments = client.scrape_post_comments(
            post_urls=post_urls,
            results_limit=args.comments,
        )
        logger.info(f"Scraped {len(all_comments)} total comments")

        if not all_comments:
            logger.error("No comments found on seed posts")
            return 1

        # Step 3: Bundle comments by user
        logger.info("Step 3: Bundling comments by username...")
        user_comments = bundle_seed_comments(
            comments=all_comments,
            max_comments_per_user=args.max_comments_per_user,
            deduplicate=True,
        )
        logger.info(f"Found {len(user_comments)} unique commenters")

        if not user_comments:
            logger.error("No valid comments found after bundling")
            return 1

        # Step 4: Sample usernames
        logger.info(f"Step 4: Sampling {args.sample} usernames...")
        sampled_users = sample_usernames(
            user_comments=user_comments,
            sample_size=args.sample,
        )
        logger.info(f"Sampled {len(sampled_users)} users")

        # Step 5: Enrich with captions
        logger.info(f"Step 5: Enriching {len(sampled_users)} users with captions...")
        bundles, skipped = enrich_with_captions(
            client=client,
            sampled_usernames=sampled_users,
            user_comments=user_comments,
            user_posts_limit=args.user_posts,
        )

        # Step 6: Write output
        logger.info(f"Step 6: Writing output to {args.out}...")

        # Ensure output only contains required fields
        output_bundles = [
            {
                "username": b["username"],
                "comments": b["comments"],
                "captions": b["captions"],
            }
            for b in bundles
        ]

        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_bundles, f, indent=2, ensure_ascii=False)

        # Final summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Seed posts scraped:      {len(post_urls)}")
        logger.info(f"Comments scraped:        {len(all_comments)}")
        logger.info(f"Unique commenters:       {len(user_comments)}")
        logger.info(f"Users sampled:           {len(sampled_users)}")
        logger.info(f"Skipped (private/unavl): {skipped}")
        logger.info(f"Bundles written:         {len(output_bundles)}")
        logger.info(f"Output file:             {args.out}")
        logger.info("=" * 60)

        if not output_bundles:
            logger.warning("No bundles were created - all users may be private")
            return 1

        logger.info("Done!")
        return 0

    except ApifyError as e:
        logger.error(f"Apify API error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
