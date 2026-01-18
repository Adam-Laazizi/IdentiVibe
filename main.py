from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from Identify.Scrapers.Youtube.youTubeScraper import YouTubeScraper

app = FastAPI()

# Enable CORS so your React website can talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/scrape/youtube/{handle}")
async def get_youtube_data(handle: str):
    try:
        # Initialize your scraper with the handle from the URL
        api_key = os.getenv("YOUTUBE_API_KEY")
        scraper = YouTubeScraper(api_key=api_key, target=handle)

        # Run your existing logic
        payload = scraper.get_payload()
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))