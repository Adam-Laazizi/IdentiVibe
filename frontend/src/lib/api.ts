import type { ResolveSourcesResponse } from "../types/sources";

const API_BASE_URL = "https://mongodbbackend-l0tv.onrender.com";

export async function resolveSources(
  query: string
): Promise<ResolveSourcesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search/resolve-sources`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error: ${response.status} ${errorText}`);
  }

  return (await response.json()) as ResolveSourcesResponse;
}
