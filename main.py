import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Existing Scraper
from Identify.Scrapers.Youtube.youTubeScraper import YouTubeScraper
# New Imports from test.py logic
from Identify.Gemini.gemini import GeminiEnricher
from Identify.Gemini.nanoBanana import NanoBananaGenerator

load_dotenv()

app = FastAPI()

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Serve the 'Identify' folder so the website can load the generated image file
# This allows you to access the image via http://localhost:8000/static/identivibe_mascot.png
if not os.path.exists("Identify/output"):
    os.makedirs("Identify/output")
app.mount("/static", StaticFiles(directory="Identify/output"), name="static")


@app.get("/scrape/youtube/{handle}")
async def get_youtube_data(handle: str):
    try:
        # 1. Initialize Classes
        api_key = os.getenv("YOUTUBE_API_KEY")
        scraper = YouTubeScraper(api_key=api_key, target=handle, vids=3
                                 ,comments= 3)
        enricher = GeminiEnricher()
        generator = NanoBananaGenerator()

        # 2. Scrape Raw Data
        print(f"Scraping YouTube handle: @{handle}")
        raw_data = scraper.get_payload()

        # 3. Enrich with Gemini (test.py logic)
        print("Enriching data with Gemini...")
        final_result = enricher.enrich_data(raw_data)

        # 4. Generate Mascot Image with Nano Banana
        report = final_result.get("community_report", {})
        visual_id = report.get("visual_identity", {})
        mascot_prompt = visual_id.get("chibi_mascot_prompt")

        image_url = None
        if mascot_prompt:
            print(f"Generating mascot: {mascot_prompt[:50]}...")
            image_filename = f"{handle}_mascot.png"
            image_path = os.path.join("Identify/output", image_filename)

            # Generate the image
            generator.generate_and_clean(mascot_prompt, image_path)

            # Create a URL the frontend can use
            image_url = f"http://localhost:8000/static/{image_filename}"
        else:
            print("Warning: No mascot prompt found in Gemini result.")

        # 5. Return the combined data
        return {
            "analysis": final_result,
            "mascot_url": image_url,
            "archetype": report.get("overall_archetype", "Unknown")
        }

    except Exception as e:
        print(f"Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))