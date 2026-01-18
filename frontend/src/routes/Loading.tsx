import { useState, useEffect } from 'react';
import type { ResolveSourcesResponse } from '../types/sources';

type LoadingProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string, state?: ResolveSourcesResponse) => void;
};

const loadingMessages = [
  'Loading...',
  'Calculating...',
  'Drinking soda...',
  'Listening to music...',
  'Stop being impatient...',
  "It's about to happen...",
  'Almost there...',
  'Crunching numbers...',
  'Consulting the AI overlords...',
  'Brewing coffee...',
  'Analyzing vibes...',
  'Reading the matrix...',
];

export function Loading({ initialState, navigate }: LoadingProps) {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (!initialState) {
      navigate('/');
      return;
    }
  }, [initialState, navigate]);

  // Cycle through messages every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % loadingMessages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // TODO: Replace this with actual API call to your backend
  // For now, simulate loading for 8 seconds then go to results
  useEffect(() => {
    const timeout = setTimeout(() => {
      navigate('/results', initialState ?? undefined);
    }, 8000);

    return () => clearTimeout(timeout);
  }, [initialState, navigate]);

  if (!initialState) {
    return null;
  }

  return (
    <div
      className="min-h-screen relative overflow-hidden flex items-center justify-center"
      style={{
        background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b3d 100%)',
      }}
    >
      {/* Animated grid background */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(rgba(139, 92, 246, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139, 92, 246, 0.3) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'gridShift 20s linear infinite',
        }}
      ></div>

      {/* Glowing orbs */}
      <div className="absolute top-20 left-10 w-96 h-96 bg-violet-500 rounded-full opacity-20 blur-3xl animate-pulse"></div>
      <div
        className="absolute bottom-20 right-10 w-96 h-96 bg-cyan-500 rounded-full opacity-20 blur-3xl animate-pulse"
        style={{ animationDelay: '1s' }}
      ></div>
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-fuchsia-500 rounded-full opacity-30 blur-3xl animate-pulse"
        style={{ animationDelay: '0.5s' }}
      ></div>

      <div className="relative text-center">
        {/* Spinning loader */}
        <div className="mb-8 flex justify-center">
          <div className="relative">
            <div
              className="w-24 h-24 border-4 border-violet-500/30 rounded-full"
              style={{ borderTopColor: '#8b5cf6', animation: 'spin 1s linear infinite' }}
            ></div>
            <div
              className="absolute inset-2 border-4 border-cyan-500/30 rounded-full"
              style={{ borderBottomColor: '#06b6d4', animation: 'spin 1.5s linear infinite reverse' }}
            ></div>
            <div
              className="absolute inset-4 border-4 border-fuchsia-500/30 rounded-full"
              style={{ borderLeftColor: '#d946ef', animation: 'spin 2s linear infinite' }}
            ></div>
          </div>
        </div>

        {/* Loading message */}
        <h1
          className="text-4xl md:text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-400 mb-4"
          style={{
            fontFamily: '"Bebas Neue", "Impact", sans-serif',
            textShadow: '0 0 30px rgba(139, 92, 246, 0.5)',
          }}
        >
          ANALYZING DATA
        </h1>

        <p
          className="text-xl md:text-2xl text-violet-300 transition-opacity duration-500"
          style={{
            fontFamily: '"Fira Code", monospace',
            animation: 'fadeInOut 2s ease-in-out infinite',
          }}
          key={messageIndex}
        >
          {loadingMessages[messageIndex]}
        </p>

        {/* Progress dots */}
        <div className="mt-8 flex justify-center gap-2">
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="w-3 h-3 bg-violet-500 rounded-full animate-pulse"
              style={{ animationDelay: `${i * 0.2}s` }}
            ></div>
          ))}
        </div>

        {/* Query display */}
        <div className="mt-8">
          <div
            className="inline-block px-5 py-2 bg-violet-500/20 backdrop-blur-sm border border-violet-400/40 text-violet-200 rounded-xl"
            style={{
              fontFamily: '"Fira Code", monospace',
              fontSize: '0.875rem',
            }}
          >
            QUERY: <span className="text-cyan-300 font-bold">{initialState.query}</span>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes gridShift {
          0% { transform: translateY(0); }
          100% { transform: translateY(50px); }
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes fadeInOut {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
