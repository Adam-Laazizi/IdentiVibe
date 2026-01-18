from typing import Optional, Any
from pydantic import BaseModel


class ResolveSourcesRequest(BaseModel):
    query: str


class Sources(BaseModel):
    redditUrl: str
    youtubeUrl: str
    instagramUrl: str
    linkedinUrl: str


class ResolveSourcesResponse(BaseModel):
    query: str
    sources: Sources


class ScrapeRequest(BaseModel):
    sources: Sources
    impatience_score: float = 0.5  # 0.0 = patient, 1.0 = impatient
    device_id: Optional[str] = None
    mock: bool = True  # Set to False for real scraping


class ScrapeResponse(BaseModel):
    instagram: Optional[dict[str, Any]] = None
    youtube: Optional[dict[str, Any]] = None
    reddit: Optional[dict[str, Any]] = None
    linkedin: Optional[dict[str, Any]] = None
    settings_used: dict[str, int]  # Show what settings were applied
    impatience_score: float  # Echo back for debugging
    analysis: Optional[dict[str, Any]] = None  # Gemini analysis results
    total_users: Optional[int] = None  # Total users scraped across platforms


class FullAnalysisRequest(BaseModel):
    """Request for complete scrape + analysis pipeline."""
    query: str
    sources: Sources
    impatience_score: float = 0.5  # 0.0 = patient, 1.0 = impatient
    device_id: Optional[str] = None


class FullAnalysisResponse(BaseModel):
    """Response from complete scrape + analysis pipeline."""
    query: str
    platforms: dict[str, dict[str, Any]]  # Platform name -> scraped data
    total_users: int
    analysis: Optional[dict[str, Any]] = None  # Gemini analysis results
    settings_used: dict[str, int]
    impatience_score: float
