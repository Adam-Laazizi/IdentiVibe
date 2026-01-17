export type Sources = {
  redditUrl: string
  youtubeUrl: string
  instagramUrl: string
  linkedinUrl: string
}

export type ResolveSourcesResponse = {
  query: string;
  sources: Sources;
};
