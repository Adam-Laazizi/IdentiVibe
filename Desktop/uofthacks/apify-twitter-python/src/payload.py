"""
Main payload generator function
"""

from src.scraper import scrape_twitter, clean_tweets
from datetime import datetime


async def git_payload(handle, limit=100):
    """
    Main payload generator function
    Takes a Twitter handle → returns structured tweet payload
    
    Args:
        handle: Twitter username (without @)
        limit: Max tweets to fetch (default: 100)
    
    Returns:
        dict: Structured payload with tweets
    """
    # Sanitize handle
    handle = handle.lstrip("@").lower().strip()
    
    print(f"[*] Fetching tweets for @{handle}...")
    
    try:
        # Scrape raw tweets (fetch more to account for filtering losses)
        raw_tweets = scrape_twitter(handle, limit * 5)
        print(f"[*] Scraped {len(raw_tweets)} raw tweets, filtering...")
        
        # Clean and filter
        filtered = clean_tweets(raw_tweets)
        print(f"[✓] Cleaned down to {len(filtered)} tweets")
        
        # Build structured payload
        payload = {
            "handle": handle,
            "count": len(filtered),
            "raw_count": len(raw_tweets),
            "tweets": [
                {"id": i + 1, "text": text}
                for i, text in enumerate(filtered)
            ],
            "generated_at": datetime.now().isoformat(),
        }
        
        return payload
    
    except Exception as e:
        raise Exception(f"Failed to fetch tweets for @{handle}: {str(e)}")


async def git_payload_batch(handles, limit=100):
    """
    Batch fetch payloads for multiple handles
    
    Args:
        handles: List of Twitter usernames
        limit: Max tweets per user
    
    Returns:
        dict: Map of handle → payload
    """
    results = {}
    
    for handle in handles:
        try:
            results[handle] = await git_payload(handle, limit)
        except Exception as e:
            print(f"[ERROR] {handle}: {str(e)}")
            results[handle] = {"error": str(e)}
    
    return results
