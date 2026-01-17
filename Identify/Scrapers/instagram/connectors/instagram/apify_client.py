"""
Minimal Apify HTTP wrapper for Instagram scraping.

Provides:
- run_actor(actor_id, input) -> run_id
- wait_for_run(run_id) -> dataset_id
- get_dataset_items(dataset_id) -> list
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

APIFY_BASE_URL = "https://api.apify.com/v2"

# Default actor ID - using the main Instagram scraper
# The individual scrapers (profile, post, comment) have been consolidated
# Use ~ separator for API calls (apify~instagram-scraper)
ACTOR_INSTAGRAM_SCRAPER = "apify~instagram-scraper"


class ApifyError(Exception):
    """Base exception for Apify API errors."""
    pass


class ApifyRateLimitError(ApifyError):
    """Rate limit exceeded."""
    pass


class ApifyClient:
    """
    Minimal Apify HTTP client with retry logic and local caching.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = "./cache",
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        timeout: int = 300,
    ):
        """
        Initialize the Apify client.

        Args:
            token: Apify API token. Defaults to APIFY_TOKEN env var.
            cache_dir: Directory for caching intermediate results.
            max_retries: Maximum number of retries for failed requests.
            backoff_factor: Backoff factor for retries.
            timeout: Request timeout in seconds.
        """
        self.token = token or os.environ.get("APIFY_TOKEN")
        if not self.token:
            raise ApifyError("APIFY_TOKEN environment variable not set")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
        }

    def _get_params(self, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get request parameters with token."""
        params: Dict[str, Any] = {"token": self.token}
        if extra_params:
            params.update(extra_params)
        return params

    def _cache_key(self, actor_id: str, input_data: Dict[str, Any]) -> str:
        """Generate a cache key from actor ID and input."""
        content = json.dumps({"actor": actor_id, "input": input_data}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cached(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached dataset items if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.debug(f"Cache hit for key {cache_key}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache file: {e}")
        return None

    def _save_cache(self, cache_key: str, items: List[Dict[str, Any]]) -> None:
        """Save dataset items to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(items, f)
            logger.debug(f"Cached {len(items)} items with key {cache_key}")
        except IOError as e:
            logger.warning(f"Failed to write cache file: {e}")

    def run_actor(
        self,
        actor_id: str,
        input_data: Dict[str, Any],
        use_cache: bool = True,
    ) -> str:
        """
        Start an actor run and return the run ID.

        Args:
            actor_id: The actor ID (e.g., "apify/instagram-profile-scraper")
            input_data: Input parameters for the actor
            use_cache: Whether to check cache before running

        Returns:
            The run ID
        """
        # Check cache first
        if use_cache:
            cache_key = self._cache_key(actor_id, input_data)
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.info(f"Using cached results for {actor_id}")
                return f"cached:{cache_key}"

        url = f"{APIFY_BASE_URL}/acts/{actor_id}/runs"

        logger.info(f"Starting actor {actor_id}")
        logger.debug(f"Input: {json.dumps(input_data, indent=2)}")

        response = self.session.post(
            url,
            headers=self._get_headers(),
            params=self._get_params(),
            json=input_data,
            timeout=self.timeout,
        )

        if response.status_code == 429:
            raise ApifyRateLimitError("Rate limit exceeded")

        response.raise_for_status()
        data = response.json()
        run_id = data["data"]["id"]
        logger.info(f"Started run {run_id}")
        return run_id

    def wait_for_run(
        self,
        run_id: str,
        poll_interval: int = 5,
        max_wait: int = 600,
    ) -> str:
        """
        Wait for an actor run to complete and return the dataset ID.

        Args:
            run_id: The run ID to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            The default dataset ID
        """
        # Handle cached runs
        if run_id.startswith("cached:"):
            return run_id

        url = f"{APIFY_BASE_URL}/actor-runs/{run_id}"
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise ApifyError(f"Run {run_id} timed out after {max_wait}s")

            response = self.session.get(
                url,
                headers=self._get_headers(),
                params=self._get_params(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]

            status = data["status"]
            logger.debug(f"Run {run_id} status: {status}")

            if status == "SUCCEEDED":
                dataset_id = data["defaultDatasetId"]
                logger.info(f"Run {run_id} completed, dataset: {dataset_id}")
                return dataset_id
            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                raise ApifyError(f"Run {run_id} ended with status: {status}")

            time.sleep(poll_interval)

    def get_dataset_items(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get items from a dataset.

        Args:
            dataset_id: The dataset ID or cached key
            limit: Maximum items to retrieve
            offset: Starting offset

        Returns:
            List of dataset items
        """
        # Handle cached datasets
        if dataset_id.startswith("cached:"):
            cache_key = dataset_id.split(":", 1)[1]
            cached = self._get_cached(cache_key)
            if cached is not None:
                if limit:
                    return cached[offset : offset + limit]
                return cached[offset:]
            raise ApifyError(f"Cache miss for key {cache_key}")

        url = f"{APIFY_BASE_URL}/datasets/{dataset_id}/items"
        extra_params: Dict[str, Any] = {"format": "json"}
        if limit:
            extra_params["limit"] = limit
        if offset:
            extra_params["offset"] = offset

        response = self.session.get(
            url,
            headers=self._get_headers(),
            params=self._get_params(extra_params),
            timeout=self.timeout,
        )
        response.raise_for_status()
        items = response.json()
        logger.info(f"Retrieved {len(items)} items from dataset {dataset_id}")
        return items

    def run_and_get_items(
        self,
        actor_id: str,
        input_data: Dict[str, Any],
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to run an actor and get all dataset items.

        Args:
            actor_id: The actor ID
            input_data: Input parameters
            use_cache: Whether to use caching

        Returns:
            List of dataset items
        """
        cache_key = self._cache_key(actor_id, input_data)

        # Check cache first
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        run_id = self.run_actor(actor_id, input_data, use_cache=False)
        dataset_id = self.wait_for_run(run_id)
        items = self.get_dataset_items(dataset_id)

        # Cache the results
        if use_cache:
            self._save_cache(cache_key, items)

        return items

    def scrape_profile_posts(
        self,
        username: str,
        results_limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Scrape posts from an Instagram profile.

        Args:
            username: Instagram username
            results_limit: Maximum number of posts to retrieve

        Returns:
            List of post data
        """
        input_data = {
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "posts",
            "resultsLimit": results_limit,
        }
        return self.run_and_get_items(ACTOR_INSTAGRAM_SCRAPER, input_data)

    def scrape_post_comments(
        self,
        post_urls: List[str],
        results_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Scrape comments from Instagram posts.

        Args:
            post_urls: List of post URLs to scrape
            results_limit: Maximum comments per post

        Returns:
            List of comment data
        """
        input_data = {
            "directUrls": post_urls,
            "resultsType": "comments",
            "resultsLimit": results_limit,
        }
        return self.run_and_get_items(ACTOR_INSTAGRAM_SCRAPER, input_data)

    def scrape_user_posts(
        self,
        username: str,
        results_limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Scrape a user's latest posts (for captions).

        Args:
            username: Instagram username
            results_limit: Maximum posts to retrieve

        Returns:
            List of post data with captions
        """
        input_data = {
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "posts",
            "resultsLimit": results_limit,
        }
        return self.run_and_get_items(ACTOR_INSTAGRAM_SCRAPER, input_data)
