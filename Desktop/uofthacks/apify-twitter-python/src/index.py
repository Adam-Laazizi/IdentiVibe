"""
Main entry point for Python version
"""

import asyncio
import sys
import json
import os
from pathlib import Path

# Handle imports from both parent and src directories
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.payload import git_payload
from src.analysis import classify_tweets


OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_analysis(analysis):
    """Pretty print Gemini analysis results"""
    print("\n" + "=" * 70)
    print("IDENTITY ANALYSIS (via Gemini)")
    print("=" * 70 + "\n")
    
    print(f"📌 Primary Identity: {analysis.get('identity', 'N/A')}")
    print(f"   Confidence: {int((analysis.get('confidence', 0) or 0) * 100)}%")
    
    sentiment = analysis.get('sentiment', 0.5) or 0.5
    emoji = "😊" if sentiment > 0.6 else "😐"
    print(f"\n{emoji} Sentiment: {int(sentiment * 100)}% positive")
    
    print(f"\n💬 Communication Tone: {analysis.get('tone', 'N/A')}")
    
    if analysis.get('categories'):
        print(f"\n🏷️  Categories:")
        for cat in analysis['categories']:
            print(f"   • {cat}")
    
    if analysis.get('interests'):
        print(f"\n💡 Key Interests:")
        for interest in analysis['interests']:
            print(f"   • {interest}")
    
    if analysis.get('primary_topics'):
        print(f"\n🎯 Primary Topics:")
        for topic in analysis['primary_topics']:
            print(f"   • {topic}")
    
    if analysis.get('summary'):
        print(f"\n📝 Summary:\n{analysis['summary']}")
    
    print("\n" + "=" * 70 + "\n")


async def main():
    """Main entry point"""
    # Get username from command-line arg or default to MrBeast
    handle = sys.argv[1] if len(sys.argv) > 1 else "MrBeast"
    
    try:
        # Get tweet payload
        payload = await git_payload(handle, 500)
        
        print("\n" + "=" * 70)
        print(f"TWEETS FOR @{handle.upper()}")
        print("=" * 70)
        print(json.dumps(payload, indent=2))
        
        # Save tweets payload to file
        tweets_file = OUTPUT_DIR / f"{handle}_tweets.json"
        with open(tweets_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[✓] Tweets payload saved to: {tweets_file}")
        
        # Optional: Classify with Gemini
        if payload.get("count", 0) > 0:
            print(f"\n[*] Classifying {payload['count']} tweets with Gemini (gemini-1.5-flash)...")
            
            try:
                tweets = [t["text"] for t in payload["tweets"]]
                analysis = classify_tweets(tweets)
                print_analysis(analysis)
                
                # Save analysis to file
                analysis_json = {
                    "handle": handle,
                    "tweet_count": payload["count"],
                    "timestamp": payload["generated_at"],
                    "analysis": analysis,
                }
                
                analysis_file = OUTPUT_DIR / f"{handle}_analysis.json"
                with open(analysis_file, "w") as f:
                    json.dump(analysis_json, f, indent=2)
                print(f"[✓] Analysis saved to: {analysis_file}")
            
            except Exception as e:
                print(f"[!] Gemini classification skipped: {str(e)}")
    
    except Exception as e:
        print(f"[FATAL] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
