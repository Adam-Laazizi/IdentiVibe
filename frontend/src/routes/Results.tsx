import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import type { ResolveSourcesResponse } from '../types/sources';

type ResultsProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string) => void;
};

// Define the interface for the data coming from your Python backend
interface ScraperResult {
  analysis: {
    community_report: {
      overall_archetype: string;
      visual_identity: {
        chibi_mascot_prompt: string;
      }
    }
  };
  mascot_url: string;
  archetype: string;
}

export function Results({ initialState, navigate }: ResultsProps) {
  const location = useLocation();

  // Extract the data we passed from Sources.tsx
  const pythonData = (location.state as any)?.analysisResult as ScraperResult;

  useEffect(() => {
    if (!initialState) {
      navigate('/');
    }
  }, [initialState, navigate]);

  if (!initialState) {
    return null;
  }

  return (
    <div className="min-h-screen relative overflow-hidden" style={{
      background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b3d 100%)'
    }}>
      {/* Animated grid background */}
      <div className="absolute inset-0 opacity-20" style={{
        backgroundImage: `
          linear-gradient(rgba(139, 92, 246, 0.3) 1px, transparent 1px),
          linear-gradient(90deg, rgba(139, 92, 246, 0.3) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px',
        animation: 'gridShift 20s linear infinite'
      }}></div>

      <div className="relative max-w-4xl mx-auto py-8 px-4">
        <h1 className="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-400 mb-8" style={{
          fontFamily: '"Bebas Neue", "Impact", sans-serif',
          textShadow: '0 0 30px rgba(139, 92, 246, 0.5)'
        }}>
          ANALYSIS COMPLETE
        </h1>

        <div className="space-y-6">
          {/* 1. AI Archetype Result */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
            <div className="relative bg-slate-900/80 backdrop-blur-sm border border-violet-500/30 rounded-2xl p-6 hover:border-violet-400/50 transition-all duration-300">
              <h2 className="text-2xl font-bold text-violet-100 mb-3 tracking-wide" style={{
                fontFamily: '"Orbitron", sans-serif'
              }}>
                COMMUNITY ARCHETYPE
              </h2>
              <div className="flex items-center gap-3">
                <div className={`w-2 h-2 ${pythonData ? 'bg-green-400' : 'bg-cyan-400'} rounded-full animate-pulse`}></div>
                <p className="text-2xl font-black text-cyan-300 uppercase tracking-tighter" style={{
                  fontFamily: '"Fira Code", monospace'
                }}>
                  {pythonData?.archetype || "// Processing Archetype..."}
                </p>
              </div>
            </div>
          </div>

          {/* 2. AI Generated Mascot Image */}
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-fuchsia-600 to-cyan-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
            <div className="relative bg-slate-900/80 backdrop-blur-sm border border-violet-500/30 rounded-2xl p-6 hover:border-violet-400/50 transition-all duration-300">
              <h2 className="text-2xl font-bold text-violet-100 mb-3 tracking-wide" style={{
                fontFamily: '"Orbitron", sans-serif'
              }}>
                VISUAL IDENTITY (MASCOT)
              </h2>
              {pythonData?.mascot_url ? (
                <div className="mt-4 flex flex-col items-center">
                  <img
                    src={pythonData.mascot_url}
                    alt="Community Mascot"
                    className="w-64 h-64 rounded-2xl border-2 border-cyan-500/50 shadow-2xl shadow-cyan-500/20 mb-4"
                  />
                  <p className="text-xs text-violet-300/50 font-mono italic">Generated via Gemini + Nano Banana</p>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                  <p className="text-violet-300/70 italic" style={{ fontFamily: '"Fira Code", monospace' }}>
                    // Image synthesis pending...
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* 3. Debug Section: Raw JSON */}
          <div className="relative group mb-8">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 to-cyan-600 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>
            <div className="relative bg-slate-900/80 backdrop-blur-sm border border-violet-500/30 rounded-2xl p-6">
              <h2 className="text-2xl font-bold text-violet-100 mb-4 tracking-wide" style={{
                fontFamily: '"Orbitron", sans-serif'
              }}>
                RAW DATA BUNDLE
              </h2>
              <div className="bg-slate-950/80 backdrop-blur-sm border border-cyan-500/30 rounded-xl p-4 max-h-60 overflow-auto">
                <pre className="text-xs text-cyan-100/70" style={{
                  fontFamily: '"Fira Code", "Courier New", monospace'
                }}>
                  {JSON.stringify(pythonData || initialState, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 flex justify-center">
          <button
            onClick={() => navigate('/')}
            className="group px-8 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-bold tracking-wider transition-all duration-300 transform hover:scale-105 shadow-lg shadow-violet-500/50"
            style={{ fontFamily: '"Orbitron", sans-serif', textTransform: 'uppercase' }}
          >
            ← New Search
          </button>
        </div>
      </div>

      <style>{`
        @keyframes gridShift {
          0% { transform: translateY(0); }
          100% { transform: translateY(50px); }
        }
      `}</style>
    </div>
  );
}