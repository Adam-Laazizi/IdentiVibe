const BASE_URL =
  (import.meta as any).env?.VITE_MONGODB_API_BASE_URL ||
  'https://mongodbbackend-l0tv.onrender.com';

async function readJsonOrThrow(res: Response, label: string) {
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const msg =
      (data && (data.error || data.message)) ||
      `${label} failed: ${res.status} ${res.statusText}`;
    throw new Error(msg);
  }

  return data;
}

export async function saveSearch(payload: {
  userId: string;
  query: string;
  platforms?: any[];
  geminiResult: any;
}) {
  const res = await fetch(`${BASE_URL}/api/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  return readJsonOrThrow(res, 'POST /api/search');
}

export async function getHistory(userId: string, limit = 20) {
  const res = await fetch(
    `${BASE_URL}/api/history/${encodeURIComponent(userId)}?limit=${limit}`
  );

  return readJsonOrThrow(res, 'GET /api/history');
}

export async function getSearchById(id: string) {
  const res = await fetch(`${BASE_URL}/api/search/${encodeURIComponent(id)}`);

  return readJsonOrThrow(res, 'GET /api/search/:id');
}
