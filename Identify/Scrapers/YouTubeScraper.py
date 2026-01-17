import os
import json
from typing import List, Dict, Any
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Ensure the import path matches your project structure
from Identify.Scrapers.SocialScraper import SocialScraper

load_dotenv()


class YouTubeScraper(SocialScraper):
    """Handles all direct interactions with the YouTube Data API."""

    def __init__(self, api_key: str):
        self.service = build('youtube', 'v3', developerKey=api_key)

    def get_channel_id(self, handle: str) -> str:
        res = self.service.search().list(
            q=handle,
            type='channel',
            part='id'
        ).execute()

        if not res.get('items'):
            raise ValueError(f"Could not find channel for handle: {handle}")

        return res['items'][0]['id']['channelId']

    def get_popular_video_ids(self, channel_id: str, limit: int = 5) -> List[
        str]:
        res = self.service.search().list(
            channelId=channel_id,
            part="id",
            order="viewCount",
            type="video",
            maxResults=limit
        ).execute()
        return [item['id']['videoId'] for item in res.get('items', [])]

    def get_video_comments(self, video_id: str, count: int = 5) -> List[Dict]:
        res = self.service.commentThreads().list(
            videoId=video_id,
            part="snippet",
            maxResults=count,
            textFormat="plainText"
        ).execute()
        return res.get('items', [])

    def get_user_topics(self, author_ids: List[str]) -> Dict[str, List[str]]:
        """Fetches topicCategories for a batch of users."""
        if not author_ids:
            return {}

        res = self.service.channels().list(
            part="topicDetails",
            id=",".join(author_ids)
        ).execute()

        topic_map = {}
        for item in res.get('items', []):
            raw_topics = item.get('topicDetails', {}).get('topicCategories', [])
            # Clean Wikipedia URLs into readable labels
            topic_map[item['id']] = [t.split('/')[-1].replace('_', ' ') for t in
                                     raw_topics]
        return topic_map

    def get_payload(self, target: str) -> dict:
        """
        Implementation of the SocialScraper interface.
        Note: We extract the key from our current service to maintain state.
        """
        yt_api_key = getattr(self.service, '_developerKey',
                             os.getenv("YOUTUBE_API_KEY"))

        # Inverting dependency: passing the scraper instance to the aggregator
        aggregator = DataAggregator(self)

        # We pass explicit limits here to prevent the '300 users' bug
        return aggregator.build_payload(target, vid_limit=5, comm_limit=5)


class DataAggregator:
    """Orchestrates the data flow to build the final JSON payload."""

    def __init__(self, client: YouTubeScraper):
        self.client = client

    def build_payload(self, handle: str, vid_limit: int = 5,
                      comm_limit: int = 5) -> Dict:
        channel_id = self.client.get_channel_id(handle)
        video_ids = self.client.get_popular_video_ids(channel_id,
                                                      limit=vid_limit)

        community_data = {
            "channel_handle": handle,
            "total_users_scanned": 0,
            "users": []
        }
        temp_users = {}

        for v_id in video_ids:
            comments = self.client.get_video_comments(v_id, count=comm_limit)

            # Extract author IDs for this specific video's batch
            author_ids = [
                c['snippet']['topLevelComment']['snippet']['authorChannelId'][
                    'value']
                for c in comments
            ]

            # Batch fetch topics (saves quota units)
            topic_lookup = self.client.get_user_topics(author_ids)

            for comment in comments:
                snippet = comment['snippet']['topLevelComment']['snippet']
                a_id = snippet['authorChannelId']['value']

                if a_id not in temp_users:
                    temp_users[a_id] = {
                        "author_id": a_id,
                        "display_name": snippet['authorDisplayName'],
                        "topics": topic_lookup.get(a_id, []),
                        "comments": [snippet['textDisplay']]
                    }
                else:
                    # Deduplication: add comment to existing user if they appeared before
                    temp_users[a_id]["comments"].append(snippet['textDisplay'])

        # Finalize the dictionary into a list for JSON export
        user_list = list(temp_users.values())
        community_data["users"] = user_list
        community_data["total_users_scanned"] = len(user_list)

        return community_data


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize the Scraper (Worker)
    api_key = os.getenv("YOUTUBE_API_KEY")
    scraper = YouTubeScraper(api_key)

    # 2. Call the Interface Method
    # This now strictly respects the 3-video / 20-comment limit
    try:
        final_payload = scraper.get_payload('@GameTheory')

        # 3. Output to local JSON
        with open("community_payload.json", "w") as f:
            json.dump(final_payload, f, indent=4)

        print(
            f"Success! Collected {final_payload['total_users_scanned']} unique users.")

    except Exception as e:
        print(f"An error occurred: {e}")