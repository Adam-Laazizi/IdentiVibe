import { useEffect } from 'react';
// REMOVED 'useLocation' to prevent the crash
import type { ResolveSourcesResponse } from '../types/sources';

type ResultsProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string) => void;
};

// Interface matching your Python backend output
interface ScraperResult {
  analysis: any;
  mascot_url: string;
  archetype: string;
}

export function Results({ initialState, navigate }: ResultsProps) {
  // 1. FIX: Read data directly from props, NOT useLocation()
  // We treat 'initialState' as the data carrier
  const state = initialState as any;

  // Robust check: handles both direct data and nested 'state' wrapper if Sources.tsx wasn't perfectly updated
  const pythonData = (state?.analysisResult || state?.state?.analysisResult) as ScraperResult;

  // 2. Check if we have data
  const hasData = !!pythonData;

  useEffect(() => {
    if (!hasData) {
      console.warn("No data in initialState. You might need to re-run the search.");
    }
  }, [hasData]);

  // 3. Fallback UI if data is missing
  if (!hasData) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-slate-900 text-white font-mono">
        <h1 className="text-2xl mb-4 text-red-400">⚠ No Analysis Data Received</h1>
        <p className="mb-6 text-slate-400 text-sm">The results page received no data from the backend.</p>
        <button
          onClick={() => navigate('/')}
          className="px-6 py-3 bg-violet-600 rounded-xl font-bold hover:bg-violet-500 transition-all"
        >
          Return Home
        </button>
        {/* Debug Info */}
        <div className="mt-8 p-4 bg-black/50 rounded text-xs text-slate-500">
          <p>Debug: initialState is {initialState ? 'Present' : 'Null'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative overflow-hidden" style={{
      background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b3d 100%)'
    }}>
      {/* Background Grid */}
      <div className="absolute inset-0 opacity-20" style={{
        backgroundImage: `linear-gradient(rgba(139, 92, 246, 0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(139, 92, 246, 0.3) 1px, transparent 1px)`,
        backgroundSize: '50px 50px'
      }}></div>

      <div className="relative max-w-4xl mx-auto py-8 px-4 text-white">
        <h1 className="text-5xl font-black mb-8 tracking-tighter uppercase italic text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-cyan-400">
          ANALYSIS COMPLETE
        </h1>

        <div className="grid gap-6">
          {/* COMMUNITY ARCHETYPE */}
          <div className="bg-slate-900/80 p-8 rounded-3xl border border-violet-500/30">
            <h2 className="text-sm font-bold text-violet-400 mb-2 uppercase tracking-widest">Community Archetype</h2>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-[0_0_10px_#4ade80]"></div>
              <p className="text-4xl font-black text-cyan-400">
                {pythonData.archetype || "Unknown Archetype"}
              </p>
            </div>
          </div>

          {/* MASCOT DISPLAY */}
          <div className="bg-slate-900/80 p-8 rounded-3xl border border-violet-500/30 flex flex-col items-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-violet-500 to-transparent opacity-50"></div>

            <h2 className="text-sm font-bold text-violet-400 mb-6 uppercase tracking-widest">Visual Identity</h2>

            {pythonData.mascot_url ? (
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-cyan-400 to-violet-600 rounded-2xl blur opacity-30 group-hover:opacity-75 transition duration-500"></div>
                <img
                  src={pythonData.mascot_url}
                  alt="Mascot"
                  className="relative w-80 h-80 rounded-2xl shadow-2xl shadow-cyan-500/20 border-2 border-white/10"
                />
              </div>
            ) : (
              <div className="w-80 h-80 flex items-center justify-center bg-slate-950/50 rounded-2xl border border-white/5">
                <p className="text-violet-300 italic animate-pulse">Mascot image loading...</p>
              </div>
            )}
          </div>

          {/* Raw Data Debug (Hidden by default, helpful for verification) */}
          <details className="bg-slate-950/50 p-4 rounded-xl border border-white/10 group">
            <summary className="cursor-pointer text-xs text-violet-300 group-hover:text-cyan-300 transition-colors">View Raw Data Payload</summary>
            <pre className="text-xs text-cyan-100/70 mt-2 overflow-auto max-h-40 font-mono">
              {JSON.stringify(pythonData, null, 2)}
            </pre>
          </details>
        </div>

        <button
          onClick={() => navigate('/')}
          className="mt-8 px-10 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-xl font-bold hover:scale-105 transition-transform shadow-lg shadow-violet-500/25"
        >
          START NEW ANALYSIS
        </button>
      </div>
    </div>
  );
}