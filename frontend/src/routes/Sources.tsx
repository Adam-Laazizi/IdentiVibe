import { useState, useEffect } from 'react';
import type { Sources, ResolveSourcesResponse } from '../types/sources';
import { saveSearch } from '../lib/mongodbApi';

type SourcesProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string, state?: ResolveSourcesResponse) => void;
};

export function Sources({ initialState, navigate }: SourcesProps) {
  const [sources, setSources] = useState<Sources | null>(null);
  const [query, setQuery] = useState('');

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');

  useEffect(() => {
    if (!initialState) {
      navigate('/');
      return;
    }
    setQuery(initialState.query);
    setSources(initialState.sources);
  }, [initialState, navigate]);

  const updateSource = (platform: keyof Sources, value: string) => {
    if (!sources) return;
    setSources({ ...sources, [platform]: value });
  };

  const handleSubmit = async () => {
    if (!sources) return;

    setSaving(true);
    setSaveError('');

    try {
      // 1. EXTRACT HANDLE: Convert URL (https://youtube.com/@tech) to handle (@tech)
      const ytInput = sources.youtubeUrl || '';
      const handle = ytInput.includes('@') ? ytInput.split('@')[1].split('/')[0] : ytInput;

      if (!handle) throw new Error('Valid YouTube handle required');

      // 2. TRIGGER SCRAPER: Call the FastAPI main.py bridge
      console.log(`Requesting Python Scrape for: @${handle}`);
      const scraperResponse = await fetch(`http://localhost:8000/scrape/youtube/${handle}`);

      if (!scraperResponse.ok) throw new Error('Python backend (main.py) is not running');
      const pythonScrapedData = await scraperResponse.json();

      // 3. MERGED INTENT: Save to MongoDB including the new scraped data
      const platformMap: Record<keyof Sources, string> = {
        redditUrl: 'reddit',
        youtubeUrl: 'youtube',
        linkedinUrl: 'linkedin',
        instagramUrl: 'instagram',
      };

      const platforms = (Object.keys(platformMap) as (keyof Sources)[])
        .filter((k) => (sources[k] || '').trim().length > 0)
        .map((k) => platformMap[k]);

      await saveSearch({
        userId: 'user123',
        query,
        platforms,
        geminiResult: {
          query,
          sources,
          scrapedData: pythonScrapedData, // Merging scraper results into the DB record
          timestamp: new Date().toISOString()
        },
      });

      // 4. NAVIGATION: Pass data to Results page using 'as any' to bypass strict linting
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      navigate('/results', {
        query,
        sources,
        analysisResult: pythonScrapedData
      } as any);

    } catch (e) {
      console.error("Submission Error:", e);
      setSaveError(e instanceof Error ? e.message : 'Failed to execute analysis');
    } finally {
      setSaving(false);
    }
  };

  if (!sources) return null;

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b3d 100%)' }}>
      {/* Background and UI Elements */}
      <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'linear-gradient(rgba(139, 92, 246, 0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(139, 92, 246, 0.3) 1px, transparent 1px)', backgroundSize: '50px 50px' }}></div>

      <div className="relative max-w-3xl mx-auto py-8 px-4">
        <h1 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-cyan-400 mb-8" style={{ fontFamily: '"Bebas Neue", sans-serif' }}>
          CONFIGURE SOURCES
        </h1>

        <div className="space-y-5 mb-8">
          {/* YouTube Input Box */}
          <div className="relative bg-slate-900/80 border border-violet-500/30 rounded-2xl p-6">
            <label className="block text-xs font-bold text-violet-300 uppercase mb-4">// YouTube Channel</label>
            <input
              type="text"
              value={sources.youtubeUrl}
              onChange={(e) => updateSource('youtubeUrl', e.target.value)}
              placeholder="https://www.youtube.com/@handle"
              className="w-full px-5 py-3 bg-slate-950/60 border border-red-500/30 text-violet-100 rounded-xl focus:border-red-400/60 outline-none"
            />
          </div>

          {/* Instagram Input Box */}
          <div className="relative bg-slate-900/80 border border-violet-500/30 rounded-2xl p-6">
            <label className="block text-xs font-bold text-violet-300 uppercase mb-4">// Instagram Profile</label>
            <input
              type="text"
              value={sources.instagramUrl}
              onChange={(e) => updateSource('instagramUrl', e.target.value)}
              placeholder="https://www.instagram.com/handle"
              className="w-full px-5 py-3 bg-slate-950/60 border border-pink-500/30 text-violet-100 rounded-xl focus:border-pink-400/60 outline-none"
            />
          </div>
        </div>

        {saveError && (
          <div className="mb-4 p-4 bg-red-950/50 border border-red-500/50 rounded-xl text-red-300 font-mono">
            ⚠ ERROR: {saveError}
          </div>
        )}

        <div className="flex gap-4">
          <button onClick={() => navigate('/')} className="px-6 py-3 text-violet-300 border border-violet-500/30 rounded-xl">← Back</button>
          <button
            onClick={handleSubmit}
            disabled={saving}
            className="flex-1 px-8 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white rounded-xl font-bold tracking-wider hover:scale-105 transition-all disabled:opacity-50"
          >
            {saving ? 'ANALYZING...' : 'Execute Analysis →'}
          </button>
        </div>
      </div>
    </div>
  );
}