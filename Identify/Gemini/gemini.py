import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiEnricher:
    def __init__(self):
        # Initialize the new 2026 SDK Client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = "gemini-2.0-flash"

        # Define safety settings as a class attribute
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_NONE"),
        ]

    def get_system_prompt(self):
        return """
        Role: You are a Senior Data Scientist and Psychographic Profiler 
        specializing in community intelligence for 'Identivibe'.

        Task: Analyze the provided 'Identity Payload'. Generate a report 
        including individual dossiers and a global community identity.

        Constraints:
        1. Output Format: Return a valid JSON object. No conversational text.
        2. The "chibi_mascot_prompt" must always end in "On a solid, pure white 
        background #FFFFFF" to ensure the background is pure white
        3. Schema:
        {
          "community_report": {
            "overall_archetype": "string",
            "community_summary": "string",
            "user_dossiers": [
              {
                "user_id": "string",
                "identity_profile": {
                  "archetype": "string",
                  "persona_description": "string",
                  "primary_interest": "string",
                  "secondary_interests": ["string"],
                  "community_role": "string",
                  "agnostic_tags": ["string"]
                },
                "statistical_validity": {
                  "confidence_score": float,
                  "data_richness": "low/medium/high",
                  "inference_basis": ["string"],
                  "sentiment_consistency": float,
                  "signal_to_noise_ratio": float
                }
              }
            ],
            "visual_identity": {
              "chibi_mascot_prompt": "string"
            }
          }
        }

        Statistical Logic:
        - confidence_score: Higher if topics align with comment vocabulary.
        - signal_to_noise_ratio: (Technical Keywords) / (Generic Words).
        """

    def enrich_data(self, raw_payload: dict):
        # Removed test_mode slice; processing all users in the payload
        users_to_process = raw_payload.get('users', [])

        print(
            f"Identivibe Intelligence: Processing {len(users_to_process)} "
            f"total users...")

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"Analyze this payload: "
                         f"{json.dumps(users_to_process)}",
                config=types.GenerateContentConfig(
                    system_instruction=self.get_system_prompt(),
                    response_mime_type="application/json",
                    safety_settings=self.safety_settings,
                    temperature=0.7
                )
            )

            if response.text:
                return json.loads(response.text)

            print("  [!] Model returned an empty response.")
            return {}

        except Exception as e:
            print(f"  [X] Enrichment Error: {e}")
            return {}