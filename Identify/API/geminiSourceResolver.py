import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

from .models import Sources

from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # Identify/ (adjust if needed)
load_dotenv(ROOT / ".env")


class GeminiSourceResolver:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing in environment/.env")

        genai.configure(api_key=api_key)
        # Use a known-good model name for the deprecated SDK
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")


    def get_system_prompt(self) -> str:
        return (
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "redditUrl": "string",\n'
            '  "youtubeUrl": "string",\n'
            '  "instagramUrl": "string",\n'
            '  "linkedinUrl": "string"\n'
            "}\n\n"
            "Rules:\n"
            "- Use fully qualified URLs\n"
            "- Prefer official or most widely recognized accounts\n"
            "- No markdown, no explanations, JSON only\n"
        )

    def resolve_sources(self, query: str) -> Sources:
        prompt = f"{self.get_system_prompt()}\nEntity: {query}"

        # Keep output small/fast and reduce non-JSON chatter
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 256,
            },
        )

        raw = getattr(resp, "text", None)
        if not raw:
            raise RuntimeError("Gemini returned empty response text")

        # SAFE JSON parse: handle cases where Gemini wraps JSON with extra text
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"Gemini did not return JSON. Raw: {raw[:300]}")
            data = json.loads(raw[start:end + 1])

        return Sources(
            redditUrl=data.get("redditUrl", ""),
            youtubeUrl=data.get("youtubeUrl", ""),
            instagramUrl=data.get("instagramUrl", ""),
            linkedinUrl=data.get("linkedinUrl", ""),
        )
