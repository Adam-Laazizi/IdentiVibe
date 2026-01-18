import React, { useState } from 'react';
import { resolveSources } from '../lib/api';
import type { ResolveSourcesResponse } from '../types/sources';

type HomeProps = {
  navigate: (path: string, state?: ResolveSourcesResponse) => void;
};

export function Home({ navigate }: HomeProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError('');

    try {
      const response = await resolveSources(trimmed);
      navigate('/sources', response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resolve sources');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading) {
      handleSubmit();
    }
  };

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

      {/* Glowing orbs */}
      <div className="absolute top-20 left-10 w-96 h-96 bg-violet-500 rounded-full opacity-20 blur-3xl animate-pulse"></div>
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-cyan-500 rounded-full opacity-20 blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>

      <div className="relative flex items-center justify-center min-h-screen px-4">
        <div className="w-full max-w-3xl">
          <div className="text-center mb-16 space-y-6">
            <h1 className="text-7xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-400 animate-pulse" style={{
              fontFamily: '"Bebas Neue", "Impact", sans-serif',
              textShadow: '0 0 40px rgba(139, 92, 246, 0.5)',
              animationDuration: '3s'
            }}>
              IDENTIVIBE
            </h1>
            <p className="text-xl text-violet-200 tracking-wide" style={{
              fontFamily: '"Courier New", monospace',
              textTransform: 'uppercase',
              letterSpacing: '0.3em'
            }}>
              &gt; Verified Social Intelligence_
            </p>
          </div>

          <div className="space-y-6">
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-violet-600 to-cyan-600 rounded-2xl blur opacity-30 group-hover:opacity-60 transition duration-500"></div>
              <div className="relative flex gap-3 bg-slate-900/80 backdrop-blur-sm rounded-2xl p-2 border border-violet-500/30">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                  placeholder="// Enter target identity..."
                  className="flex-1 px-6 py-4 bg-transparent text-violet-100 placeholder-violet-400/50 focus:outline-none disabled:opacity-50 text-lg"
                  style={{ fontFamily: '"Fira Code", "Courier New", monospace' }}
                  aria-label="Search query"
                />
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="px-8 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-bold tracking-wider transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center gap-3 shadow-lg shadow-violet-500/50"
                  style={{ fontFamily: '"Orbitron", sans-serif', textTransform: 'uppercase' }}
                >
                  {loading && (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  )}
                  {loading ? 'Scanning' : 'Execute'}
                </button>
              </div>
            </div>

            {error && (
              <div className="relative overflow-hidden p-5 bg-red-950/50 backdrop-blur-sm border border-red-500/50 rounded-xl animate-pulse">
                <div className="absolute inset-0 bg-gradient-to-r from-red-500/10 to-transparent"></div>
                <p className="relative text-red-300 font-medium" style={{ fontFamily: '"Fira Code", monospace' }}>
                  ⚠ ERROR: {error}
                </p>
              </div>
            )}

            <div className="flex justify-center pt-4">
              <button
                onClick={() => navigate('/history')}
                className="group px-6 py-3 text-violet-300 hover:text-violet-100 border border-violet-500/30 hover:border-violet-400/60 rounded-xl transition-all duration-300 backdrop-blur-sm hover:bg-violet-500/10"
                style={{ fontFamily: '"Fira Code", monospace' }}
              >
                <span className="inline-block transition-transform duration-300 group-hover:translate-x-1">
                  → Access History Archive
                </span>
              </button>
            </div>
          </div>
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