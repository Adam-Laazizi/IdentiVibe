"""
ScraperMerger - Orchestrates all social media scrapers and sends merged output to Gemini.

This is the main entry point for the web application scraping pipeline.
It coordinates YouTube, Instagram, Reddit, and LinkedIn scrapers, merges their outputs,
and sends the combined payload to Google Gemini for analysis.

Usage (Web Application):
    merger = ScraperMerger(config)
    result = await merger.scrape_and_analyze(sources, settings)

Usage (Programmatic):
    merger = ScraperMerger(config)
    merged = merger.merge_payloads(payloads)
    analysis = merger.analyze_with_gemini(merged)
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Configuration for all scrapers."""
    # API Keys
    youtube_api_key: Optional[str] = None
    apify_token: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Scraping settings (can be adjusted based on impatience score)
    posts_limit: int = 10
    comments_limit: int = 150
    sample_size: int = 250
    user_posts_limit: int = 10
    max_users: int = 30

    # Execution settings
    timeout_seconds: int = 120
    max_workers: int = 4
    cache_dir: str = "./cache"

    @classmethod
    def from_env(cls, settings: Optional[Dict[str, int]] = None) -> "ScraperConfig":
        """Create config from environment variables with optional settings override."""
        config = cls(
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
            apify_token=os.getenv("APIFY_TOKEN"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
        )
        if settings:
            config.posts_limit = settings.get("posts", config.posts_limit)
            config.comments_limit = settings.get("comments", config.comments_limit)
            config.sample_size = settings.get("sample", config.sample_size)
            config.user_posts_limit = settings.get("user_posts", config.user_posts_limit)
        return config


@dataclass
class ScrapedPlatformData:
    """Container for scraped data from a single platform."""
    platform: str
    handle: str
    users: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "handle": self.handle,
            "users": self.users,
            "user_count": len(self.users),
            "metadata": self.metadata,
            "error": self.error,
            "scraped_at": self.scraped_at,
        }

    @property
    def is_empty(self) -> bool:
        return len(self.users) == 0 and self.error is None


@dataclass
class MergedPayload:
    """Container for merged data from all platforms."""
    query: str
    platforms: Dict[str, ScrapedPlatformData] = field(default_factory=dict)
    total_users: int = 0
    merged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
            "total_users": self.total_users,
            "merged_at": self.merged_at,
            "analysis": self.analysis,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ScraperMerger:
    """
    Orchestrates all social media scrapers, merges outputs, and sends to Gemini for analysis.

    This class serves as the parent controller that:
    1. Initializes and manages individual platform scrapers
    2. Runs scrapers in parallel for efficiency
    3. Merges all outputs into a unified JSON structure
    4. Handles empty/failed scrapes gracefully
    5. Sends merged data to Gemini for community analysis
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the ScraperMerger with configuration.

        Args:
            config: ScraperConfig instance. If None, loads from environment.
        """
        self.config = config or ScraperConfig.from_env()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._gemini_model = None

    def _init_gemini(self) -> None:
        """Lazily initialize Gemini model."""
        if self._gemini_model is None:
            api_key = self.config.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for analysis")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

    # ==================== Handle Extraction ====================

    @staticmethod
    def extract_youtube_handle(url: str) -> str:
        """Extract YouTube handle from various URL formats."""
        if not url:
            return ""
        url = url.strip()

        # Handle @handle format
        if url.startswith("@"):
            return url.split("/")[0].split("?")[0]

        # Parse URL
        if "youtube.com" in url or "youtu.be" in url:
            if "/@" in url:
                return "@" + url.split("/@")[1].split("/")[0].split("?")[0]
            if "/channel/" in url:
                return url.split("/channel/")[1].split("/")[0].split("?")[0]
            if "/c/" in url:
                return "@" + url.split("/c/")[1].split("/")[0].split("?")[0]
            if "/user/" in url:
                return "@" + url.split("/user/")[1].split("/")[0].split("?")[0]

        # Fallback: treat as handle
        clean = url.replace("https://", "").replace("http://", "")
        clean = clean.replace("www.youtube.com/", "").replace("youtube.com/", "")
        handle = clean.split("/")[0].split("?")[0]
        return f"@{handle}" if handle and not handle.startswith("@") else handle

    @staticmethod
    def extract_instagram_handle(url: str) -> str:
        """Extract Instagram handle from URL or handle string."""
        if not url:
            return ""
        url = url.strip()

        if url.startswith("@"):
            return url[1:].split("/")[0].split("?")[0]

        if "instagram.com" in url:
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if path:
                return path.split("/")[0]

        # Fallback: treat as handle
        return url.lstrip("@").split("/")[0].split("?")[0]

    @staticmethod
    def extract_reddit_subreddit(url: str) -> str:
        """Extract subreddit name from Reddit URL."""
        if not url:
            return ""
        url = url.strip()

        if url.startswith("r/"):
            return url[2:].split("/")[0].split("?")[0]

        if "reddit.com" in url:
            if "/r/" in url:
                return url.split("/r/")[1].split("/")[0].split("?")[0]

        # Fallback
        return url.replace("r/", "").split("/")[0].split("?")[0]

    @staticmethod
    def extract_linkedin_profile(url: str) -> str:
        """Extract LinkedIn profile identifier from URL."""
        if not url:
            return ""
        url = url.strip()

        if "linkedin.com" in url:
            if "/in/" in url:
                return url.split("/in/")[1].split("/")[0].split("?")[0]
            if "/company/" in url:
                return url.split("/company/")[1].split("/")[0].split("?")[0]

        # Fallback
        return url

    # ==================== Individual Scrapers ====================

    def _scrape_youtube(self, handle: str) -> ScrapedPlatformData:
        """Run YouTube scraper and return standardized data."""
        result = ScrapedPlatformData(platform="youtube", handle=handle)

        if not handle:
            result.error = "No handle provided"
            return result

        if not self.config.youtube_api_key:
            result.error = "YouTube API key not configured"
            return result

        try:
            # Import here to avoid circular imports
            from Identify.Scrapers.Youtube.youTubeScraper import YouTubeScraper

            scraper = YouTubeScraper(
                api_key=self.config.youtube_api_key,
                target=handle,
                vids=min(self.config.posts_limit, 10),  # YouTube quota-aware limit
                comments=min(self.config.comments_limit, 100),
            )

            payload = scraper.get_payload()

            # Standardize output format
            result.users = [
                {
                    "username": u.get("display_name", u.get("author_id", "")),
                    "user_id": u.get("author_id"),
                    "comments": u.get("comments", []),
                    "topics": u.get("topics", []),
                }
                for u in payload.get("users", [])
            ]
            result.metadata = {
                "channel_handle": payload.get("channel_handle"),
                "total_users_scanned": payload.get("total_users_scanned", len(result.users)),
            }

            logger.info(f"YouTube scrape complete: {len(result.users)} users from {handle}")

        except Exception as e:
            logger.exception(f"YouTube scrape error for {handle}")
            result.error = str(e)

        return result

    def _scrape_instagram(self, handle: str) -> ScrapedPlatformData:
        """Run Instagram scraper and return standardized data."""
        result = ScrapedPlatformData(platform="instagram", handle=handle)

        if not handle:
            result.error = "No handle provided"
            return result

        if not self.config.apify_token:
            result.error = "Apify token not configured"
            return result

        try:
            from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper

            settings = {
                "posts": self.config.posts_limit,
                "comments": self.config.comments_limit,
                "sample": self.config.sample_size,
                "user_posts": self.config.user_posts_limit,
                "max_comments_per_user": 50,
                "cache_dir": self.config.cache_dir,
            }

            scraper = InstagramScraper(
                apify_token=self.config.apify_token,
                target=handle,
                settings=settings,
            )

            payload = scraper.get_payload()

            # Standardize output format
            result.users = [
                {
                    "username": u.get("username", ""),
                    "comments": u.get("comments", []),
                    "captions": u.get("captions", []),
                }
                for u in payload.get("users", [])
            ]
            result.metadata = {
                "seed_handle": payload.get("seed_handle"),
            }

            logger.info(f"Instagram scrape complete: {len(result.users)} users from {handle}")

        except Exception as e:
            logger.exception(f"Instagram scrape error for {handle}")
            result.error = str(e)

        return result

    def _scrape_reddit(self, subreddit: str) -> ScrapedPlatformData:
        """Run Reddit scraper and return standardized data."""
        result = ScrapedPlatformData(platform="reddit", handle=subreddit)

        if not subreddit:
            result.error = "No subreddit provided"
            return result

        try:
            from Identify.Scrapers.Reddit.RedditScraper import RedditScraper

            scraper = RedditScraper()

            payload = scraper.get_payload(
                target_subreddit=subreddit,
                post_sample=min(self.config.posts_limit, 20),
                comment_sample_per_post=40,
                max_users=self.config.max_users,
                user_comment_limit=60,
                user_post_limit=30,
            )

            # Standardize output format
            result.users = [
                {
                    "username": u.get("username", ""),
                    "comments": u.get("global_activity", {}).get("recent_comments", []),
                    "posts": u.get("global_activity", {}).get("recent_posts", []),
                    "subreddit_activity": u.get("global_activity", {}).get("subreddit_histogram", []),
                }
                for u in payload.get("users", [])
            ]
            result.metadata = {
                "source_subreddit": payload.get("source_subreddit"),
                "note": payload.get("note"),
            }

            logger.info(f"Reddit scrape complete: {len(result.users)} users from r/{subreddit}")

        except Exception as e:
            logger.exception(f"Reddit scrape error for r/{subreddit}")
            result.error = str(e)

        return result

    def _scrape_linkedin(self, profile: str) -> ScrapedPlatformData:
        """Run LinkedIn scraper and return standardized data."""
        result = ScrapedPlatformData(platform="linkedin", handle=profile)

        if not profile:
            result.error = "No profile provided"
            return result

        if not self.config.apify_token:
            result.error = "Apify token not configured"
            return result

        try:
            from Identify.Scrapers.linkedin.scrape_linkedin import LinkedInScraper

            scraper = LinkedInScraper(
                api_key=self.config.apify_token,
                profile_or_username=profile,
            )

            # Note: LinkedIn scraper uses git_payload() not get_payload() - handle both
            if hasattr(scraper, "get_payload"):
                payload = scraper.get_payload()
            else:
                payload = scraper.git_payload()

            # LinkedIn returns a list, standardize to our format
            if isinstance(payload, list) and payload:
                profile_data = payload[0]
                result.users = [
                    {
                        "username": profile_data.get("name", ""),
                        "profile_url": profile_data.get("profile_url", ""),
                        "profile_data": profile_data.get("profile_data", {}),
                        "posts": profile_data.get("posts", []),
                    }
                ]
                result.metadata = {
                    "profile_url": profile_data.get("profile_url"),
                }

            logger.info(f"LinkedIn scrape complete: {len(result.users)} profiles for {profile}")

        except Exception as e:
            logger.exception(f"LinkedIn scrape error for {profile}")
            result.error = str(e)

        return result

    # ==================== Main Orchestration ====================

    def scrape_all(
        self,
        youtube_url: str = "",
        instagram_url: str = "",
        reddit_url: str = "",
        linkedin_url: str = "",
        parallel: bool = True,
    ) -> Dict[str, ScrapedPlatformData]:
        """
        Run all scrapers and return results.

        Args:
            youtube_url: YouTube channel URL or handle
            instagram_url: Instagram profile URL or handle
            reddit_url: Reddit subreddit URL or name
            linkedin_url: LinkedIn profile URL or username
            parallel: Whether to run scrapers in parallel

        Returns:
            Dictionary mapping platform names to ScrapedPlatformData
        """
        # Extract handles from URLs
        handles = {
            "youtube": self.extract_youtube_handle(youtube_url),
            "instagram": self.extract_instagram_handle(instagram_url),
            "reddit": self.extract_reddit_subreddit(reddit_url),
            "linkedin": self.extract_linkedin_profile(linkedin_url),
        }

        scrapers = {
            "youtube": (self._scrape_youtube, handles["youtube"]),
            "instagram": (self._scrape_instagram, handles["instagram"]),
            "reddit": (self._scrape_reddit, handles["reddit"]),
            "linkedin": (self._scrape_linkedin, handles["linkedin"]),
        }

        results: Dict[str, ScrapedPlatformData] = {}

        if parallel:
            # Run scrapers in parallel using ThreadPoolExecutor
            futures = {}
            for platform, (scraper_fn, handle) in scrapers.items():
                if handle:  # Only run if handle is provided
                    future = self._executor.submit(scraper_fn, handle)
                    futures[future] = platform
                else:
                    # Create empty result for missing platforms
                    results[platform] = ScrapedPlatformData(
                        platform=platform,
                        handle="",
                        error="No URL provided"
                    )

            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                platform = futures[future]
                try:
                    results[platform] = future.result()
                except Exception as e:
                    logger.exception(f"Scraper {platform} failed")
                    results[platform] = ScrapedPlatformData(
                        platform=platform,
                        handle=handles[platform],
                        error=str(e)
                    )
        else:
            # Run scrapers sequentially
            for platform, (scraper_fn, handle) in scrapers.items():
                if handle:
                    results[platform] = scraper_fn(handle)
                else:
                    results[platform] = ScrapedPlatformData(
                        platform=platform,
                        handle="",
                        error="No URL provided"
                    )

        return results

    def merge_payloads(
        self,
        query: str,
        platform_data: Dict[str, ScrapedPlatformData],
    ) -> MergedPayload:
        """
        Merge scraped data from all platforms into a unified payload.

        Args:
            query: Original search query
            platform_data: Dictionary of platform data

        Returns:
            MergedPayload containing all data
        """
        merged = MergedPayload(query=query, platforms=platform_data)

        # Calculate total users across all platforms
        merged.total_users = sum(
            len(data.users) for data in platform_data.values()
        )

        logger.info(
            f"Merged payload: {merged.total_users} total users across "
            f"{len([p for p in platform_data.values() if not p.is_empty])} platforms"
        )

        return merged

    def analyze_with_gemini(self, merged: MergedPayload) -> Dict[str, Any]:
        """
        Send merged payload to Gemini for community analysis.

        Args:
            merged: MergedPayload to analyze

        Returns:
            Analysis results from Gemini
        """
        self._init_gemini()

        # Build analysis prompt
        prompt = self._build_analysis_prompt(merged)

        try:
            response = self._gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 4096,
                },
            )

            raw = getattr(response, "text", "")

            # Parse JSON response
            try:
                analysis = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end > start:
                    analysis = json.loads(raw[start:end + 1])
                else:
                    analysis = {"raw_response": raw, "parse_error": True}

            merged.analysis = analysis
            return analysis

        except Exception as e:
            logger.exception("Gemini analysis failed")
            error_result = {"error": str(e)}
            merged.analysis = error_result
            return error_result

    def _build_analysis_prompt(self, merged: MergedPayload) -> str:
        """Build the analysis prompt for Gemini."""
        # Prepare condensed data for analysis
        platform_summaries = []
        all_comments = []
        all_topics = []

        for platform, data in merged.platforms.items():
            if data.error or data.is_empty:
                continue

            platform_summaries.append({
                "platform": platform,
                "user_count": len(data.users),
                "handle": data.handle,
            })

            for user in data.users[:50]:  # Limit users per platform to avoid token limits
                comments = user.get("comments", [])[:10]
                captions = user.get("captions", [])[:5]
                posts = user.get("posts", [])[:5]
                topics = user.get("topics", [])

                all_comments.extend(comments)
                all_comments.extend(captions)
                all_comments.extend(posts)
                all_topics.extend(topics)

        # Limit total content to avoid token limits
        all_comments = all_comments[:200]
        all_topics = list(set(all_topics))[:50]

        prompt = f"""Analyze this community data and provide insights.

Query: {merged.query}

Platform Summary:
{json.dumps(platform_summaries, indent=2)}

Topics Found: {', '.join(all_topics) if all_topics else 'None extracted'}

Sample Content (comments, posts, captions):
{json.dumps(all_comments[:100], indent=2, ensure_ascii=False)}

Return a JSON object with this structure:
{{
    "community_profile": {{
        "primary_interests": ["list", "of", "main", "interests"],
        "secondary_interests": ["list", "of", "secondary", "interests"],
        "demographic_signals": ["list", "of", "demographic", "indicators"],
        "engagement_style": "description of how the community engages",
        "content_themes": ["major", "themes", "in", "content"]
    }},
    "sentiment_analysis": {{
        "overall_sentiment": "positive/neutral/negative/mixed",
        "sentiment_breakdown": {{
            "positive_percentage": 0,
            "neutral_percentage": 0,
            "negative_percentage": 0
        }},
        "key_emotions": ["list", "of", "detected", "emotions"]
    }},
    "platform_insights": {{
        "cross_platform_patterns": ["patterns", "across", "platforms"],
        "platform_specific_behaviors": {{
            "youtube": "behavior description or null",
            "instagram": "behavior description or null",
            "reddit": "behavior description or null",
            "linkedin": "behavior description or null"
        }}
    }},
    "recommendations": {{
        "content_suggestions": ["content", "ideas"],
        "engagement_opportunities": ["ways", "to", "engage"],
        "potential_partnerships": ["partnership", "ideas"]
    }},
    "confidence_score": 0.0
}}

Provide only valid JSON, no markdown or explanations."""

        return prompt

    async def scrape_and_analyze(
        self,
        query: str,
        youtube_url: str = "",
        instagram_url: str = "",
        reddit_url: str = "",
        linkedin_url: str = "",
        skip_analysis: bool = False,
    ) -> MergedPayload:
        """
        Complete pipeline: scrape all platforms, merge, and analyze.

        This is the main entry point for the web application.

        Args:
            query: Original search query
            youtube_url: YouTube channel URL
            instagram_url: Instagram profile URL
            reddit_url: Reddit subreddit URL
            linkedin_url: LinkedIn profile URL
            skip_analysis: If True, skip Gemini analysis

        Returns:
            MergedPayload with scraped data and analysis
        """
        # Run scraping in thread pool to not block async event loop
        loop = asyncio.get_event_loop()

        platform_data = await loop.run_in_executor(
            self._executor,
            lambda: self.scrape_all(
                youtube_url=youtube_url,
                instagram_url=instagram_url,
                reddit_url=reddit_url,
                linkedin_url=linkedin_url,
                parallel=True,
            )
        )

        # Merge payloads
        merged = self.merge_payloads(query, platform_data)

        # Analyze with Gemini (always, even if empty - per requirements)
        if not skip_analysis:
            await loop.run_in_executor(
                self._executor,
                lambda: self.analyze_with_gemini(merged)
            )

        return merged

    def scrape_and_analyze_sync(
        self,
        query: str,
        youtube_url: str = "",
        instagram_url: str = "",
        reddit_url: str = "",
        linkedin_url: str = "",
        skip_analysis: bool = False,
    ) -> MergedPayload:
        """
        Synchronous version of scrape_and_analyze for non-async contexts.
        """
        platform_data = self.scrape_all(
            youtube_url=youtube_url,
            instagram_url=instagram_url,
            reddit_url=reddit_url,
            linkedin_url=linkedin_url,
            parallel=True,
        )

        merged = self.merge_payloads(query, platform_data)

        if not skip_analysis:
            self.analyze_with_gemini(merged)

        return merged

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)


# ==================== Convenience Functions ====================

def create_merger_from_settings(settings: Dict[str, int]) -> ScraperMerger:
    """
    Create a ScraperMerger with settings adjusted for impatience score.

    Args:
        settings: Dictionary with posts, comments, sample, user_posts

    Returns:
        Configured ScraperMerger instance
    """
    config = ScraperConfig.from_env(settings)
    return ScraperMerger(config)


def get_impatience_settings(impatience_score: float) -> Dict[str, int]:
    """
    Map impatience score to scraper settings.

    Args:
        impatience_score: Float from 0.0 (patient) to 1.0 (impatient)

    Returns:
        Settings dictionary
    """
    if impatience_score > 0.7:
        # Impatient user - fast settings
        return {"posts": 3, "comments": 50, "sample": 50, "user_posts": 3}
    elif impatience_score > 0.4:
        # Moderate - balanced settings
        return {"posts": 5, "comments": 100, "sample": 100, "user_posts": 5}
    else:
        # Patient user - thorough settings
        return {"posts": 10, "comments": 150, "sample": 250, "user_posts": 10}


# ==================== CLI Support ====================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python scraperMerger.py <query> [--youtube URL] [--instagram URL] [--reddit URL] [--linkedin URL]")
        print("\nExample:")
        print('  python scraperMerger.py "UofT" --youtube "@uoft" --instagram "uoft" --reddit "UofT"')
        sys.exit(1)

    # Parse arguments
    query = sys.argv[1]
    args = sys.argv[2:]

    urls = {
        "youtube_url": "",
        "instagram_url": "",
        "reddit_url": "",
        "linkedin_url": "",
    }

    i = 0
    while i < len(args):
        if args[i] == "--youtube" and i + 1 < len(args):
            urls["youtube_url"] = args[i + 1]
            i += 2
        elif args[i] == "--instagram" and i + 1 < len(args):
            urls["instagram_url"] = args[i + 1]
            i += 2
        elif args[i] == "--reddit" and i + 1 < len(args):
            urls["reddit_url"] = args[i + 1]
            i += 2
        elif args[i] == "--linkedin" and i + 1 < len(args):
            urls["linkedin_url"] = args[i + 1]
            i += 2
        else:
            i += 1

    # Run the merger
    merger = ScraperMerger()
    result = merger.scrape_and_analyze_sync(query, **urls)

    # Output
    output_file = "merged_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.to_json())

    print(f"\nSuccess! Merged {result.total_users} users across platforms.")
    print(f"Output saved to: {output_file}")

    if result.analysis:
        print("\nAnalysis Summary:")
        if "community_profile" in result.analysis:
            interests = result.analysis["community_profile"].get("primary_interests", [])
            print(f"  Primary Interests: {', '.join(interests[:5])}")
        if "sentiment_analysis" in result.analysis:
            sentiment = result.analysis["sentiment_analysis"].get("overall_sentiment", "unknown")
            print(f"  Overall Sentiment: {sentiment}")
