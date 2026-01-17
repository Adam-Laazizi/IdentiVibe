# linkedin

LinkedIn scraping helpers using Apify actors.

## Setup

- Ensure `APIFY_TOKEN` is set in your environment or `.env`.
- Optional:
  - `LINKEDIN_PROFILE_ACTOR` (default: `dev_fusion/Linkedin-Profile-Scraper`)
  - `LINKEDIN_POSTS_ACTOR` (no default; if set, script will attempt to fetch posts)

## Usage

```bash
python linkedin/scrape_linkedin.py https://www.linkedin.com/in/williamhgates
python linkedin/scrape_linkedin.py urls.txt
```

If you pass a `.txt` file, each line is treated as a LinkedIn profile URL.

## Output

Results are saved to `linkedin/output/` as JSON files.
