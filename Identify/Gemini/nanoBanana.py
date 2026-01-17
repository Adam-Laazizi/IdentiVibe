import os
import io
from PIL import Image
from google import genai
from google.genai import types
from rembg import remove, new_session
from dotenv import load_dotenv

load_dotenv()


class NanoBananaGenerator:
    def __init__(self):
        # Initialize GenAI Client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Using the Pro model for higher-fidelity 'Thinking' outputs
        self.model_id = "gemini-3-pro-image-preview"
        # Pre-load rembg session for speed
        self.rembg_session = new_session()

    def generate_and_clean(self, prompt: str,
                           output_filename: str = "mascot.png"):
        print(f"🍌 Nano Banana Pro: Generating image for prompt...")

        try:
            # We removed thinking_config because Nano Banana Pro
            # handles reasoning internally for image composition.
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="1:1"),
                    # We can still keep temperature high for creative variety
                    temperature=1.0
                )
            )

            image_part = next((p for p in response.parts if p.inline_data),
                              None)
            if not image_part:
                print("  [X] Error: No image data in response.")
                return None

            # Process the image with PIL and rembg
            raw_image = Image.open(io.BytesIO(image_part.inline_data.data))

            print("  [+] Image generated. Stripping background with rembg...")
            cleaned_image = remove(raw_image, session=self.rembg_session)
            cleaned_image.save(output_filename, "PNG")

            print(f"  [---] Success! Mascot saved to {output_filename}")
            return output_filename

        except Exception as e:
            print(f"  [X] Generator Error: {e}")
            return None


if __name__ == "__main__":
    generator = NanoBananaGenerator()
    test_prompt = (
        "A cute chibi that is fat and banana shaped. Wearing a night gown."
    )
    generator.generate_and_clean(test_prompt, "identivibe_mascot.png")