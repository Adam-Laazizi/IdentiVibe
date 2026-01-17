import os
import json
from typing import List, Dict, Any
from googleapiclient.discovery import build
from dotenv import load_dotenv

from Identify.Scrapers.SocialScraper import SocialScraper

load_dotenv()


class YouTubeScraper(SocialScraper):
    """Handles all direct interactions with the YouTube Data API."""

    def __init__(self, api_key: str):
        self.service = build('youtube', 'v3', developerKey=api_key)

    def get_channel_id(self, handle: str) -> str:
        res = self.service.search().list(q=handle, type='channel',
                                         part='id').execute()
        return res['items'][0]['id']['channelId']

    def get_popular_video_ids(self, channel_id: str, limit: int = 3) -> List[
        str]:
        res = self.service.search().list(
            channelId=channel_id, part="id", order="viewCount", type="video",
            maxResults=limit
        ).execute()
        return [item['id']['videoId'] for item in res.get('items', [])]

    def get_video_comments(self, video_id: str, count: int = 20) -> List[Dict]:
        res = self.service.commentThreads().list(
            videoId=video_id, part="snippet", maxResults=count,
            textFormat="plainText"
        ).execute()
        return res.get('items', [])

    def get_user_topics(self, author_ids: List[str]) -> Dict[str, List[str]]:
        """Fetches topicCategories for a batch of users."""
        res = self.service.channels().list(part="topicDetails",
                                           id=",".join(author_ids)).execute()
        topic_map = {}
        for item in res.get('items', []):
            raw_topics = item.get('topicDetails', {}).get('topicCategories', [])
            topic_map[item['id']] = [t.split('/')[-1].replace('_', ' ') for t in
                                     raw_topics]
        return topic_map

    """
    Pre(target): target is 
    """
    def get_payload(self, target: str) -> dict:
        yt_api_key = os.getenv("YOUTUBE_API_KEY")
        client = YouTubeScraper(yt_api_key)
        aggregator = DataAggregator(client)

        # Produce the "Source of Truth" JSON
        return aggregator.build_payload(target)


class DataAggregator:
    """Orchestrates the data flow to build the final JSON payload."""

    def __init__(self, client: YouTubeScraper):
        self.client = client

    def build_payload(self, handle: str) -> Dict:
        channel_id = self.client.get_channel_id(handle)
        video_ids = self.client.get_popular_video_ids(channel_id)

        community_data = {"channel_handle": handle, "users": []}
        temp_users = {}

        for v_id in video_ids:
            comments = self.client.get_video_comments(v_id)
            author_ids = [
                c['snippet']['topLevelComment']['snippet']['authorChannelId'][
                    'value'] for c in comments]

            # Batch fetch topics to save quota
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
                    temp_users[a_id]["comments"].append(snippet['textDisplay'])

        community_data["users"] = list(temp_users.values())
        return community_data


# --- Main Execution ---
if __name__ == "__main__":
    yt_api_key = os.getenv("YOUTUBE_API_KEY")
    client = YouTubeScraper(yt_api_key)
    aggregator = DataAggregator(client)

    # Produce the "Source of Truth" JSON
    final_payload = aggregator.build_payload("@LinusTechTips")

    with open("community_payload.json", "w") as f:
        json.dump(final_payload, f, indent=4)