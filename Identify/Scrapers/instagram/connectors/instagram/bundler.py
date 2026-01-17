"""
Instagram data bundling and enrichment logic.

Provides:
- bundle_seed_comments(seed_posts_and_comments) -> dict[username, comments]
- enrich_with_captions(client, sampled_usernames, user_posts_limit) -> list[bundles]
- is_private_or_unavailable(user_scrape_result) -> bool
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .apify_client import ApifyClient

logger = logging.getLogger(__name__)


def is_private_or_unavailable(user_scrape_result: List[Dict[str, Any]]) -> bool:
    """
    Check if a user's account is private or unavailable.

    A user is considered private/unavailable if:
    - The scrape result is empty (no items returned)
    - Any item has isPrivate=true
    - The result indicates an error or forbidden status

    Args:
        user_scrape_result: List of items from scraping user's posts

    Returns:
        True if user is private or unavailable, False otherwise
    """
    if not user_scrape_result:
        logger.debug("User result empty - marking as unavailable")
        return True

    for item in user_scrape_result:
        # Check explicit private flag
        if item.get("isPrivate", False):
            logger.debug("User marked as private")
            return True

        # Check for error indicators
        if item.get("error"):
            logger.debug(f"User has error: {item.get('error')}")
            return True

        # Check for forbidden/private in error messages
        error_msg = str(item.get("errorMessage", "")).lower()
        if "private" in error_msg or "forbidden" in error_msg:
            logger.debug(f"User error message indicates private: {error_msg}")
            return True

    return False


def bundle_seed_comments(
    comments: List[Dict[str, Any]],
    max_comments_per_user: int = 50,
    deduplicate: bool = True,
) -> Dict[str, List[str]]:
    """
    Bundle comments by username from seed account's posts.

    Args:
        comments: List of comment data from Apify
        max_comments_per_user: Maximum comments to keep per user
        deduplicate: Whether to deduplicate comment text per user

    Returns:
        Dictionary mapping username -> list of comment texts
    """
    user_comments: Dict[str, List[str]] = defaultdict(list)

    for comment in comments:
        # Extract username - handle different field names from Apify
        username = (
            comment.get("ownerUsername")
            or comment.get("username")
            or comment.get("owner", {}).get("username")
        )

        if not username:
            logger.debug(f"Skipping comment without username: {comment.get('id', 'unknown')}")
            continue

        # Extract comment text - handle different field names
        text = (
            comment.get("text")
            or comment.get("body")
            or comment.get("content")
            or ""
        ).strip()

        if not text:
            continue

        # Skip if already at max for this user
        if len(user_comments[username]) >= max_comments_per_user:
            continue

        # Deduplicate if enabled
        if deduplicate and text in user_comments[username]:
            continue

        user_comments[username].append(text)

    logger.info(f"Bundled comments from {len(user_comments)} unique users")
    return dict(user_comments)


def sample_usernames(
    user_comments: Dict[str, List[str]],
    sample_size: int,
    min_comments: int = 1,
) -> List[str]:
    """
    Randomly sample usernames from the comment pool.

    Args:
        user_comments: Dictionary of username -> comments
        sample_size: Number of users to sample
        min_comments: Minimum comments required to be sampled

    Returns:
        List of sampled usernames
    """
    # Filter users with minimum comments
    eligible_users = [
        username
        for username, comments in user_comments.items()
        if len(comments) >= min_comments
    ]

    # Sample or return all if sample_size exceeds eligible
    actual_sample = min(sample_size, len(eligible_users))
    sampled = random.sample(eligible_users, actual_sample)

    logger.info(
        f"Sampled {len(sampled)} users from {len(eligible_users)} eligible "
        f"(requested {sample_size})"
    )
    return sampled


def extract_captions(posts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract caption texts from post data.

    Args:
        posts: List of post data from Apify

    Returns:
        List of caption strings
    """
    captions = []
    for post in posts:
        # Handle different field names from Apify
        caption = (
            post.get("caption")
            or post.get("text")
            or post.get("description")
            or ""
        )

        # Caption might be a string or an object with text
        if isinstance(caption, dict):
            caption = caption.get("text", "")

        caption = caption.strip() if caption else ""

        if caption:
            captions.append(caption)

    return captions


def enrich_with_captions(
    client: ApifyClient,
    sampled_usernames: List[str],
    user_comments: Dict[str, List[str]],
    user_posts_limit: int = 10,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Enrich sampled users with captions from their posts.

    For each sampled user:
    1. Scrape their latest posts
    2. Check if private/unavailable
    3. If available, extract captions and create bundle

    Args:
        client: ApifyClient instance
        sampled_usernames: List of usernames to enrich
        user_comments: Dictionary of username -> comments
        user_posts_limit: Number of posts to scrape per user

    Returns:
        Tuple of (list of bundles, number of users skipped)
    """
    bundles: List[Dict[str, Any]] = []
    skipped_count = 0

    for i, username in enumerate(sampled_usernames, 1):
        logger.info(f"Processing user {i}/{len(sampled_usernames)}: {username}")

        try:
            # Scrape user's posts
            posts = client.scrape_user_posts(username, results_limit=user_posts_limit)

            # Check if private or unavailable
            if is_private_or_unavailable(posts):
                logger.info(f"Skipping {username}: private or unavailable")
                skipped_count += 1
                continue

            # Extract captions
            captions = extract_captions(posts)

            if not captions:
                logger.info(f"Skipping {username}: no captions found")
                skipped_count += 1
                continue

            # Create bundle
            bundle = {
                "username": username,
                "comments": user_comments.get(username, []),
                "captions": captions,
            }
            bundles.append(bundle)
            logger.debug(
                f"Created bundle for {username}: "
                f"{len(bundle['comments'])} comments, {len(bundle['captions'])} captions"
            )

        except Exception as e:
            logger.warning(f"Error processing {username}: {e}")
            skipped_count += 1
            continue

    logger.info(
        f"Enrichment complete: {len(bundles)} bundles created, "
        f"{skipped_count} users skipped"
    )
    return bundles, skipped_count


def extract_post_urls(posts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract post URLs from profile scrape results.

    Args:
        posts: List of post data from profile scrape

    Returns:
        List of post URLs
    """
    urls = []
    for post in posts:
        url = (
            post.get("url")
            or post.get("postUrl")
            or post.get("shortCode") and f"https://www.instagram.com/p/{post['shortCode']}/"
        )
        if url:
            urls.append(url)
    return urls
