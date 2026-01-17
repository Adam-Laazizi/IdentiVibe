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
