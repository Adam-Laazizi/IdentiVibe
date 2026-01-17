# reddit.py (NO OAuth / NO tokens)
# Uses Reddit's public JSON endpoints. Best-effort: may be rate-limited/blocked.

import time
from typing import Dict, Any, List, Optional, Set

import requests

from Identify.Scrapers.SocialScraper import SocialScraper


class RedditScraper(SocialScraper):
    """Discover users from a subreddit, then fetch each user's global activity.

    Public endpoints used:
      - /r/{sub}/hot.json, /r/{sub}/new.json
      - /r/{sub}/comments/{post_id}.json
      - /user/{name}/comments.json
      - /user/{name}/submitted.json

    No OAuth. No API keys. If you need reliable high-volume access, use OAuth.
    """

    BASE = "https://www.reddit.com"

    def __init__(
        self,
        user_agent: str = "identify-ai/0.1 (no-oauth) by u/your_username",
        timeout_s: int = 20,
        max_retries: int = 3,
        backoff_s: float = 1.2,
        min_delay_s: float = 0.25,
    ):
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.min_delay_s = min_delay_s
        self._last_request_t = 0.0

    # ---------------- HTTP helpers ----------------
    def _sleep_if_needed(self) -> None:
        dt = time.time() - self._last_request_t
        if dt < self.min_delay_s:
            time.sleep(self.min_delay_s - dt)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.BASE}{path}"
        headers = {"User-Agent": self.user_agent}

        attempt = 0
        while True:
            attempt += 1
            self._sleep_if_needed()
            self._last_request_t = time.time()

            r = requests.get(url, params=params, headers=headers, timeout=self.timeout_s)

            # Retry-friendly statuses
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt > self.max_retries:
                    r.raise_for_status()
                time.sleep(self.backoff_s * attempt)
                continue

            # Some environments get sporadic 403 due to gating; retry a bit.
            if r.status_code in (401, 403):
                if attempt > self.max_retries:
                    r.raise_for_status()
                time.sleep(self.backoff_s * attempt)
                continue

            r.raise_for_status()
            return r.json()

    # ---------------- Subreddit discovery ----------------
    def _subreddit_posts(self, subreddit: str, sort: str, limit: int) -> List[Dict[str, Any]]:
        data = self._get(f"/r/{subreddit}/{sort}.json", {"limit": limit})
        return data.get("data", {}).get("children", [])

    def _post_comments(self, subreddit: str, post_id: str, limit: int) -> List[Dict[str, Any]]:
        data = self._get(f"/r/{subreddit}/comments/{post_id}.json", {"limit": limit})
        if isinstance(data, list) and len(data) >= 2:
            return data[1].get("data", {}).get("children", [])
        return []

    # ---------------- User global activity ----------------
    def _user_comments(self, username: str, limit: int) -> List[Dict[str, Any]]:
        data = self._get(f"/user/{username}/comments.json", {"limit": limit})
        return data.get("data", {}).get("children", [])

    def _user_posts(self, username: str, limit: int) -> List[Dict[str, Any]]:
        data = self._get(f"/user/{username}/submitted.json", {"limit": limit})
        return data.get("data", {}).get("children", [])

    # ---------------- Payload ----------------
    @staticmethod
    def _safe_author(author: Any) -> Optional[str]:
        if not author or not isinstance(author, str):
            return None
        if author.lower() == "[deleted]":
            return None
        return author

    @staticmethod
    def _top_histogram(freq: Dict[str, int], top_k: int = 12) -> List[Dict[str, Any]]:
        items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"subreddit": k, "count": v} for k, v in items]

    def get_payload(
        self,
        target_subreddit: str,
        post_sample: int = 20,
        comment_sample_per_post: int = 40,
        max_users: int = 30,
        user_comment_limit: int = 60,
        user_post_limit: int = 30,
    ) -> Dict[str, Any]:
        """Return ONLY global behaviour for users discovered from `target_subreddit`."""

        # 1) Discover users from subreddit (posts + comments)
        posts = (
            self._subreddit_posts(target_subreddit, "hot", max(1, post_sample // 2))
            + self._subreddit_posts(target_subreddit, "new", max(1, post_sample // 2))
        )

        usernames: List[str] = []
        seen: Set[str] = set()

        for p in posts:
            pdata = (p or {}).get("data", {})
            pid = pdata.get("id")

            a = self._safe_author(pdata.get("author"))
            if a and a not in seen:
                seen.add(a)
                usernames.append(a)
                if len(usernames) >= max_users:
                    break

            if not pid:
                continue

            comments = self._post_comments(target_subreddit, pid, comment_sample_per_post)
            for c in comments:
                cdata = (c or {}).get("data", {})
                au = self._safe_author(cdata.get("author"))
                if au and au not in seen:
                    seen.add(au)
                    usernames.append(au)
                    if len(usernames) >= max_users:
                        break

            if len(usernames) >= max_users:
                break

        # 2) Fetch each user's global activity
        out_users: List[Dict[str, Any]] = []

        for uname in usernames:
            user_obj: Dict[str, Any] = {
                "username": uname,
                "global_activity": {
                    "recent_comments": [],
                    "recent_posts": [],
                    "subreddit_histogram": [],
                },
            }

            try:
                comments = self._user_comments(uname, user_comment_limit)
                posts = self._user_posts(uname, user_post_limit)

                sub_freq: Dict[str, int] = {}
                recent_comments: List[str] = []
                recent_posts: List[str] = []

                for item in comments:
                    d = (item or {}).get("data", {})
                    sr = d.get("subreddit")
                    body = d.get("body")
                    if sr:
                        sub_freq[sr] = sub_freq.get(sr, 0) + 1
                    if body:
                        recent_comments.append(body)

                for item in posts:
                    d = (item or {}).get("data", {})
                    sr = d.get("subreddit")
                    title = d.get("title")
                    selftext = d.get("selftext")
                    if sr:
                        sub_freq[sr] = sub_freq.get(sr, 0) + 1
                    if title:
                        recent_posts.append(title if not selftext else f"{title}\n{selftext}")

                user_obj["global_activity"] = {
                    "recent_comments": recent_comments[:user_comment_limit],
                    "recent_posts": recent_posts[:user_post_limit],
                    "subreddit_histogram": self._top_histogram(sub_freq, top_k=12),
                }

            except Exception:
                # Best-effort: keep empty global_activity.
                pass

            out_users.append(user_obj)

        return {
            "platform": "reddit",
            "source_subreddit": target_subreddit,
            "users": out_users,
            "note": "No OAuth used. Data via public reddit.com JSON endpoints; may be rate-limited/blocked.",
        }
