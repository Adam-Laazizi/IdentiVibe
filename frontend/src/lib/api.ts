import type { ResolveSourcesResponse } from "../types/sources";

export async function resolveSources(
  query: string
): Promise<ResolveSourcesResponse> {
  const response = await fetch("http://127.0.0.1:8000/api/resolve-sources", {
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
