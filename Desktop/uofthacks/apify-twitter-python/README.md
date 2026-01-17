# apify-twitter-python

Python version of Twitter scraper using Apify + Gemini.

## Setup

```bash
cd apify-twitter-python
pip install -r requirements.txt
```

Add your credentials to `.env`:
- `APIFY_TOKEN` from https://console.apify.com
- `GEMINI_API_KEY` from https://makersuite.google.com

## Usage

```bash
python src/index.py elonmusk
python src/index.py nasa
python src/index.py mrbeast
```

## Output

Results saved to `output/` directory:
- `{username}_tweets.json` - All cleaned tweets
- `{username}_analysis.json` - Gemini classification
