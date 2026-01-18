import asyncio
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ResolveSourcesRequest,
    ResolveSourcesResponse,
    ScrapeRequest,
    ScrapeResponse,
    FullAnalysisRequest,
    FullAnalysisResponse,
)
from .geminiSourceResolver import GeminiSourceResolver
from Identify.Scrapers.scraperMerger import (
    ScraperMerger,
    ScraperConfig,
    get_impatience_settings,
    create_merger_from_settings,
)

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_resolver: Optional[GeminiSourceResolver] = None
_merger: Optional[ScraperMerger] = None


def get_resolver() -> GeminiSourceResolver:
    global _resolver
    if _resolver is None:
        _resolver = GeminiSourceResolver()
    return _resolver


def get_merger(settings: Optional[Dict[str, int]] = None) -> ScraperMerger:
    """Get or create a ScraperMerger instance with optional settings."""
    global _merger
    if settings:
        # Create new merger with specific settings
        return create_merger_from_settings(settings)
    if _merger is None:
        _merger = ScraperMerger()
    return _merger


def generate_mock_data(platform: str, handle: str) -> dict:
    """Generate fake data for testing without Apify costs."""
    return {
        "seed_handle": handle,
        "users": [
            {
                "username": f"mock_user_{i}",
                "comments": [f"Mock comment {j} from user {i}" for j in range(3)],
                "captions": [f"Mock caption {j} from user {i}" for j in range(2)],
            }
            for i in range(5)
        ],
        "_mock": True,
        "_platform": platform,
    }


@app.post("/api/resolve-sources", response_model=ResolveSourcesResponse)
def resolve_sources_endpoint(req: ResolveSourcesRequest):
    try:
        resolver = get_resolver()
        sources = resolver.resolve_sources(req.query)
        return {"query": req.query, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scrape", response_model=ScrapeResponse)
async def scrape_endpoint(req: ScrapeRequest):
    """
    Scrape social media sources with settings based on user impatience.

    Set mock=True (default) to return fake data for testing.
    Set mock=False to use real Apify scraping.
    """
    try:
        settings = get_impatience_settings(req.impatience_score)
        logger.info(
            f"Scraping with impatience={req.impatience_score:.2f}, "
            f"settings={settings}, mock={req.mock}"
        )

        result: dict = {
            "settings_used": settings,
            "impatience_score": req.impatience_score,
        }

        if req.mock:
            # Simulate some processing time (shorter for impatient users)
            delay = 2.0 if req.impatience_score > 0.7 else 4.0 if req.impatience_score > 0.4 else 6.0
            await asyncio.sleep(delay)

            # Generate mock data for each provided source
            if req.sources.instagramUrl:
                handle = req.sources.instagramUrl.rstrip("/").split("/")[-1]
                result["instagram"] = generate_mock_data("instagram", handle)

            if req.sources.youtubeUrl:
                handle = req.sources.youtubeUrl.rstrip("/").split("/")[-1]
                result["youtube"] = generate_mock_data("youtube", handle)

            if req.sources.redditUrl:
                handle = req.sources.redditUrl.rstrip("/").split("/")[-1]
                result["reddit"] = generate_mock_data("reddit", handle)

            if req.sources.linkedinUrl:
                handle = req.sources.linkedinUrl.rstrip("/").split("/")[-1]
                result["linkedin"] = generate_mock_data("linkedin", handle)

        else:
            # Real scraping using ScraperMerger
            merger = get_merger(settings)

            # Run the full scrape and analyze pipeline
            merged_result = await merger.scrape_and_analyze(
                query=req.sources.youtubeUrl or req.sources.instagramUrl or "unknown",
                youtube_url=req.sources.youtubeUrl,
                instagram_url=req.sources.instagramUrl,
                reddit_url=req.sources.redditUrl,
                linkedin_url=req.sources.linkedinUrl,
                skip_analysis=False,
            )

            # Convert platform data to response format
            for platform, data in merged_result.platforms.items():
                platform_dict = data.to_dict()
                if platform == "youtube":
                    result["youtube"] = platform_dict
                elif platform == "instagram":
                    result["instagram"] = platform_dict
                elif platform == "reddit":
                    result["reddit"] = platform_dict
                elif platform == "linkedin":
                    result["linkedin"] = platform_dict

            # Add analysis to result
            result["analysis"] = merged_result.analysis
            result["total_users"] = merged_result.total_users

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Scrape endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=FullAnalysisResponse)
async def full_analysis_endpoint(req: FullAnalysisRequest):
    """
    Complete analysis pipeline: scrape all platforms, merge data, and analyze with Gemini.

    This endpoint runs all scrapers in parallel, merges the outputs, and sends
    the combined data to Gemini for community analysis.
    """
    try:
        settings = get_impatience_settings(req.impatience_score)
        logger.info(
            f"Full analysis for query='{req.query}' with impatience={req.impatience_score:.2f}"
        )

        merger = get_merger(settings)

        # Run the complete pipeline
        result = await merger.scrape_and_analyze(
            query=req.query,
            youtube_url=req.sources.youtubeUrl,
            instagram_url=req.sources.instagramUrl,
            reddit_url=req.sources.redditUrl,
            linkedin_url=req.sources.linkedinUrl,
            skip_analysis=False,
        )

        return FullAnalysisResponse(
            query=req.query,
            platforms={k: v.to_dict() for k, v in result.platforms.items()},
            total_users=result.total_users,
            analysis=result.analysis,
            settings_used=settings,
            impatience_score=req.impatience_score,
        )

    except Exception as e:
        logger.exception("Full analysis endpoint error")
        raise HTTPException(status_code=500, detail=str(e))
