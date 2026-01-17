import os
import json
from dotenv import load_dotenv

# Import your specialized classes
from Identify.Scrapers.Youtube.YouTubeScraper import YouTubeScraper
from Identify.Gemini.gemini import GeminiEnricher
from Identify.Gemini.nanoBanana import NanoBananaGenerator

# Always load env at the very top of your entry point
load_dotenv()


def main():
    target_handle = "@Fireship"

    # 1. Initialize Workers
    # Passing the key directly from env to the constructor
    scraper = YouTubeScraper(os.getenv("YOUTUBE_API_KEY"))
    enricher = GeminiEnricher()

    print(f"--- Starting Identivibe Pipeline for {target_handle} ---")

    # 2. Execution Phase
    try:
        print("[1/2] Scraping community data...")
        raw_data = scraper.get_payload(target_handle)

        print(
            f"[2/2] Enriching {raw_data['total_users_scanned']} users with Gemini...")
        final_payload = enricher.enrich_data(raw_data)

        # 3. Persistence Phase
        filename = f"identivibe_{target_handle.replace('@', '')}.json"
        with open(filename, "w") as f:
            json.dump(final_payload, f, indent=4)

        print(f"--- Success! Data saved to {filename} ---")

    except Exception as e:
        print(f"Pipeline Failed: {e}")


# --- Main Execution (test.py) ---
if __name__ == "__main__":
    scraper = YouTubeScraper(os.getenv("YOUTUBE_API_KEY"))
    chanel = input("Please enter a youtube channel: \n@")
    raw_data = scraper.get_payload('@' + chanel)

    enricher = GeminiEnricher()
    # final_result is now a DICTIONARY, not a list
    final_result = enricher.enrich_data(raw_data)

    # 1. Access the report correctly
    report = final_result.get("community_report", {})

    # 2. Extract your specific new fields
    archetype = report.get("overall_archetype")
    mascot_prompt = report.get("visual_identity", {}).get("chibi_mascot_prompt")
    users = report.get("user_dossiers", [])

    print(f"Community Archetype: {archetype}")
    print(f"Mascot Prompt: {mascot_prompt[:50]}...")  # Show just the start
    print(f"Processed {len(users)} individual dossiers.")

    # Save the whole thing
    with open("identivibe_final.json", "w") as f:
        json.dump(final_result, f, indent=4)

    generator = NanoBananaGenerator()
    generator.generate_and_clean(mascot_prompt, "identivibe_mascot.png")

