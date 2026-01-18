// src/lib/api.ts
import type { ResolveSourcesResponse } from '../types/sources';

const MONGODB_BACKEND = 'https://mongodbbackend-l0tv.onrender.com';

async function readJsonOrThrow(res: Response) {
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const msg =
      (data && (data.error || data.message)) || `HTTP ${res.status} ${res.statusText}`;
    throw new Error(msg);
  }

  return data;
}

// POST https://mongodbbackend-l0tv.onrender.com/api/search/resolve-sources
export async function resolveSources(query: string): Promise<ResolveSourcesResponse> {
  const res = await fetch(`${MONGODB_BACKEND}/api/search/resolve-sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  // backend returns: { query, sources }
  return readJsonOrThrow(res);
}
