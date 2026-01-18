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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running", "message": "Welcome to Identivibe API"}

# Serve the 'Identify/output' folder so images are web-accessible
if not os.path.exists("Identify/output"):
    os.makedirs("Identify/output")
app.mount("/static", StaticFiles(directory="Identify/output"), name="static")

@app.get("/scrape/youtube/{handle}")
async def get_youtube_data(handle: str):
    try:
        # 1. Initialize Classes
        api_key = os.getenv("YOUTUBE_API_KEY")
        scraper = YouTubeScraper(api_key=api_key, target=handle, vids=3, comments=3)
        enricher = GeminiEnricher()
        generator = NanoBananaGenerator()

        # 2. Scrape Raw Data
        print(f"Scraping YouTube handle: @{handle}")
        raw_data = scraper.get_payload()

        # 3. Enrich with Gemini
        print("Enriching data with Gemini...")
        final_result = enricher.enrich_data(raw_data)

        # 4. Generate Mascot Image
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

            # --- PRODUCTION URL ---
            image_url = f"https://identivibe.onrender.com/static/{image_filename}"


        else:
            print("Warning: No mascot prompt found.")

        # 5. RETURN FIXED STRUCTURE
        # We nest everything in 'analysisResult' to match Results.tsx
        return {
            "analysisResult": {
                "analysis": final_result,
                "mascot_url": image_url,
                "archetype": report.get("overall_archetype", "Unknown")
            }
        }

    except Exception as e:
        print(f"Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))