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

        # MEMORY FIX: Using 'u2netp' to stay under Render's RAM limits
        self.rembg_session = new_session("u2netp")

    def generate_and_clean(self, prompt: str,
                           output_filename: str = "mascot.png"):
        print(f"🍌 Nano Banana Pro: Generating image for prompt...")

        try:
            # --- SDK FIX: Use ImageConfig instead of GenerateImageConfig ---
            response = self.client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
                config=types.ImageConfig(
                    aspect_ratio="1:1",
                    number_of_images=1,
                    include_rai_reasoning=True
                )
            )

            if not response.generated_images:
                print("  [X] Error: No image returned by Google.")
                return None

            image_bytes = response.generated_images[0].image.image_bytes

            # Process the image with PIL and rembg
            raw_image = Image.open(io.BytesIO(image_bytes))

            print("  [+] Image generated. Stripping background with rembg...")

            cleaned_image = remove(raw_image, session=self.rembg_session)

            # Ensure output directory exists
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            cleaned_image.save(output_filename, "PNG")

            print(f"  [---] Success! Mascot saved to {output_filename}")
            return output_filename

        except Exception as e:
            print(f"  [X] Generator Error: {e}")
            return None


if __name__ == "__main__":
    generator = NanoBananaGenerator()
    test_prompt = "A cute chibi mascot for a tech channel."
    generator.generate_and_clean(test_prompt, "identivibe_mascot.png")