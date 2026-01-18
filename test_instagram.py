import json
from dotenv import load_dotenv

from Identify.Scrapers.instagram.instagram_scraper import InstagramScraper
from Identify.Gemini.gemini import GeminiEnricher
from Identify.Gemini.nanoBanana import NanoBananaGenerator

load_dotenv()


def main():
    handle_input = input("Please enter an instagram handle: \n@")

    scraper = InstagramScraper()
    raw_data = scraper.get_payload(handle_input)

    if not raw_data.get("users"):
        print("No users found. Writing empty output.")
        with open("identivibe_instagram.json", "w") as f:
            json.dump(raw_data, f, indent=4)
        return

    enricher = GeminiEnricher()
    generator = NanoBananaGenerator()

    print("Enriching data with Gemini...")
    final_result = enricher.enrich_data(raw_data)

    report = final_result.get("community_report", {})
    visual_id = report.get("visual_identity", {})
    mascot_prompt = visual_id.get("chibi_mascot_prompt")

    archetype = report.get("overall_archetype", "Unknown")

    print(f"Community Archetype: {archetype}")

    if mascot_prompt:
        print(f"Mascot Prompt: {mascot_prompt[:70]}...")
        generator.generate_and_clean(mascot_prompt, "identivibe_mascot.png")
    else:
        print("Error: Could not extract chibi_mascot_prompt.")

    with open("identivibe_final.json", "w") as f:
        json.dump(final_result, f, indent=4)


if __name__ == "__main__":
    main()
