"""
Gemini AI classification
"""

import os
import json
from google.genai import Client
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def classify_tweets(tweets):
    """
    Classify tweets using Gemini AI to infer user identity
    
    Args:
        tweets: List of cleaned tweet texts
    
    Returns:
        dict: Classification analysis
    """
    if not tweets:
        raise ValueError("No tweets provided for classification")
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    
    client = Client(api_key=key)
    
    # Build tweet context
    tweet_text = "\n".join([f'{i+1}. "{t}"' for i, t in enumerate(tweets)])
    
    prompt = f"""You are a social media analyst specializing in user profiling and identity inference.

Analyze the following tweets from a Twitter user and classify their primary identity, interests, and persona.

TWEETS:
{tweet_text}

Based on these tweets, provide a JSON response with EXACTLY this structure (no markdown, no extra text):
{{
  "identity": "A 2-3 word primary identity/persona (e.g. 'Tech Entrepreneur', 'Climate Activist', 'Science Communicator')",
  "categories": ["category1", "category2", "category3"],
  "sentiment": 0.65,
  "confidence": 0.85,
  "summary": "A detailed 2-3 sentence summary of this user's online identity, primary interests, and communication style",
  "interests": ["interest1", "interest2", "interest3"],
  "tone": "professional|humorous|casual|passionate|academic",
  "primary_topics": ["topic1", "topic2"]
}}

IMPORTANT:
- sentiment must be a number between 0 (negative) and 1 (positive)
- confidence must be a number between 0 (low) and 1 (high)
- tone must be ONE of: professional, humorous, casual, passionate, academic
- Return ONLY valid JSON, no markdown or extra text
- Categories should be 3-5 unique identifiers"""
    
    try:
        print("[*] Sending tweets to Gemini (gemini-1.5-flash) for classification...")
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        parsed = json.loads(response.text)
        
        # Validate response structure
        if not all(k in parsed for k in ["identity", "categories", "sentiment"]):
            raise ValueError("Invalid response structure from Gemini")
        
        return parsed
    
    except Exception as e:
        print(f"[ERROR] Gemini classification failed: {str(e)}")
        raise
