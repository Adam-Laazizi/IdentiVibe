"""
Twitter scraper using Apify client
"""

from apify_client import ApifyClient
import os
import re
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def scrape_twitter(handle, limit=100):
    """Scrape raw tweets from Twitter via Apify"""
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_TOKEN in .env")
    
    client = ApifyClient(token)
    
    run = client.actor("quacker/twitter-scraper").call(run_input={
        "startUrls": [{"url": f"https://twitter.com/{handle}"}],
        "maxItems": limit,
    })
    
    # Get items from dataset
    dataset = client.dataset(run["defaultDatasetId"])
    items = []
    for item in dataset.iterate_items():
        items.append(item)
    
    return items


def clean_tweets(raw_tweets):
    """
    Clean and filter tweets for semantic analysis
    Removes: links, emojis, mentions, hashtags, special chars
    """
    tweets = []
    
    for tweet in raw_tweets:
        # Extract text
        text = tweet.get("text", "").strip()
        if not text:
            continue
        
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        
        # Remove @mentions and #hashtags
        text = re.sub(r"[@#]\S+", "", text)
        
        # Remove emojis and special Unicode characters
        text = re.sub(r"[\U0001F300-\U0001F9FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]", "", text)
        
        # Remove special characters, keep only letters, numbers, basic punctuation
        text = re.sub(r"[^A-Za-z0-9 .,!?'-]", "", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Filter by minimum length
        if len(text) < 20:
            continue
        
        # Must contain letters
        if not re.search(r"[A-Za-z]", text):
            continue
        
        tweets.append(text)
    
    return tweets
