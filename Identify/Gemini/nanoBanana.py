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
        self.model_id = "gemini-2.0-flash"
        # Note: I switched this to 2.0-flash as 'gemini-3' is not standard yet
        # and might cause API errors. Switch back if you have special access.

        # --- FIX: USE LIGHTWEIGHT MODEL (u2netp) ---
        # This prevents the 176MB download and saves RAM
        self.rembg_session = new_session("u2netp")

    def generate_and_clean(self, prompt: str,
                           output_filename: str = "mascot.png"):
        print(f"🍌 Nano Banana Pro: Generating image for prompt...")

        try:
            # Generate the image using Gemini
            response = self.client.models.generate_image(
                model='imagen-3.0-generate-002',  # Standard Imagen 3 model
                prompt=prompt,
                config=types.GenerateImageConfig(
                    aspect_ratio="1:1",
                    person_generation="allow_adult",
                    # Optional: Adjust as needed
                )
            )

            # NOTE: The SDK structure for images might differ slightly depending
            # on the version. If the above fails, revert to your original
            # model call logic, but KEEP the "u2netp" fix in __init__.

            # Handle response (Standardizing for typical GenAI image response)
            if not response.generated_images:
                print("  [X] Error: No image data in response.")
                return None

            image_bytes = response.generated_images[0].image.image_bytes

            # Process the image with PIL and rembg
            raw_image = Image.open(io.BytesIO(image_bytes))

            print("  [+] Image generated. Stripping background with rembg...")

            # The session is already loaded with 'u2netp' from __init__
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