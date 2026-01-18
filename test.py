import os
import json
from dotenv import load_dotenv

from Identify.Scrapers.Youtube.youTubeScraper import YouTubeScraper
from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper
from Identify.Gemini.gemini import GeminiEnricher
from Identify.Gemini.nanoBanana import NanoBananaGenerator

load_dotenv()


def main():
    # 1. Get User Input first so we can pass it to the constructor
    channel_input = input("Please enter a youtube channel: \n@")

    # 2. Initialize Classes
    # Pass BOTH arguments to the constructor as you intended
    scraper = YouTubeScraper(os.getenv("YOUTUBE_API_KEY"), channel_input, 2, 2)
    enricher = GeminiEnricher()
    generator = NanoBananaGenerator()

    # 3. Scrape and Enrich
    # Note: get_payload() no longer needs arguments
    raw_data = scraper.get_payload()

    print("Enriching data with Gemini...")
    final_result = enricher.enrich_data(raw_data)

    # 4. Data Extraction (Matches your complex JSON schema)
    report = final_result.get("community_report", {})
    visual_id = report.get("visual_identity", {})
    mascot_prompt = visual_id.get("chibi_mascot_prompt")

    archetype = report.get("overall_archetype", "Unknown")

    # 5. Output and Generation
    print(f"Community Archetype: {archetype}")

    if mascot_prompt:
        print(f"Mascot Prompt: {mascot_prompt[:70]}...")
        generator.generate_and_clean(mascot_prompt, "identivibe_mascot.png")
    else:
        print("Error: Could not extract chibi_mascot_prompt.")

    # 6. Save result
    with open("identivibe_final.json", "w") as f:
        json.dump(final_result, f, indent=4)


if __name__ == "__main__":
    main()