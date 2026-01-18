import { useState, useEffect, useRef, useCallback } from 'react';
import type { ResolveSourcesResponse } from '../types/sources';

type LoadingProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string, state?: unknown) => void;
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

// Amplitude tracking config
const AMPLITUDE_API_KEY = 'fade1179fcf08d585410d2ae5c8c46ae';
const RAGE_WINDOW_MS = 2000;
const RAGE_THRESHOLD = 7;
const STORAGE_KEY_IMPATIENCE = 'identivibe_impatience_score';
const STORAGE_KEY_DEVICE_ID = 'identivibe_device_id';

// Generate a simple device ID if none exists
function getOrCreateDeviceId(): string {
  let deviceId = localStorage.getItem(STORAGE_KEY_DEVICE_ID);
  if (!deviceId) {
    deviceId = `device_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem(STORAGE_KEY_DEVICE_ID, deviceId);
  }
  return deviceId;
}

// Get previous impatience score (default 0.5 for new users)
function getStoredImpatienceScore(): number {
  const stored = localStorage.getItem(STORAGE_KEY_IMPATIENCE);
  return stored ? parseFloat(stored) : 0.5;
}

// Save impatience score for next visit
function saveImpatienceScore(score: number): void {
  localStorage.setItem(STORAGE_KEY_IMPATIENCE, score.toString());
}

// Compute new impatience score based on behavior
function computeImpatienceScore(
  rageClicks: number,
  totalClicks: number,
  tabSwitches: number,
  waitTimeMs: number
): number {
  let score = 0;

  // Rage clicks are a strong signal
  if (rageClicks > 0) score += 0.4;

  // High click count indicates impatience
  if (totalClicks > 20) score += 0.2;
  else if (totalClicks > 10) score += 0.1;

  // Tab switches indicate distraction/impatience
  if (tabSwitches > 2) score += 0.2;
  else if (tabSwitches > 0) score += 0.1;

  // Patient if they waited without much clicking
  if (waitTimeMs > 5000 && totalClicks < 5) score -= 0.2;

  // Clamp between 0 and 1
  return Math.max(0, Math.min(1, score + 0.3)); // Base of 0.3
}

export function Loading({ initialState, navigate }: LoadingProps) {
  const [messageIndex, setMessageIndex] = useState(0);
  const [status, setStatus] = useState<'loading' | 'error'>('loading');
  const [errorMsg, setErrorMsg] = useState('');

  // Tracking refs (don't trigger re-renders)
  const clickTimesRef = useRef<number[]>([]);
  const totalClicksRef = useRef(0);
  const rageClicksRef = useRef(0);
  const tabSwitchesRef = useRef(0);
  const startTimeRef = useRef(Date.now());
  const hasCompletedRef = useRef(false);

  // Get stored values
  const deviceId = useRef(getOrCreateDeviceId());
  const previousImpatience = useRef(getStoredImpatienceScore());

  // Track clicks for rage detection
  const handleClick = useCallback(() => {
    const now = performance.now();
    totalClicksRef.current += 1;

    // Add to sliding window
    clickTimesRef.current.push(now);
    clickTimesRef.current = clickTimesRef.current.filter(
      (t) => now - t <= RAGE_WINDOW_MS
    );

    // Check for rage click
    if (clickTimesRef.current.length >= RAGE_THRESHOLD) {
      rageClicksRef.current += 1;
      console.log('🔥 Rage click detected!', {
        clicks: clickTimesRef.current.length,
        total: totalClicksRef.current,
      });
      // Reset window after rage detection
      clickTimesRef.current = [];
    }
  }, []);

  // Track tab switches
  const handleVisibilityChange = useCallback(() => {
    if (document.visibilityState === 'hidden') {
      tabSwitchesRef.current += 1;
      console.log('👀 Tab switch detected', { count: tabSwitchesRef.current });
    }
  }, []);

  // Save behavior on page unload (potential rage quit)
  const handleBeforeUnload = useCallback(() => {
    if (!hasCompletedRef.current) {
      // User left before completion - high impatience
      saveImpatienceScore(0.9);
      console.log('😤 Rage quit detected - saving high impatience');
    }
  }, []);

  // Redirect if no initial state
  useEffect(() => {
    if (!initialState) {
      navigate('/');
      return;
    }
  }, [initialState, navigate]);

  // Set up event listeners for tracking
  useEffect(() => {
    document.addEventListener('click', handleClick);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [handleClick, handleVisibilityChange, handleBeforeUnload]);

  // Cycle through messages every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % loadingMessages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Call the actual scrape API
  useEffect(() => {
    if (!initialState) return;

    const runScrape = async () => {
      try {
        console.log('🚀 Starting scrape with impatience:', previousImpatience.current);

        const response = await fetch('http://localhost:8000/api/scrape', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sources: initialState.sources,
            impatience_score: previousImpatience.current,
            device_id: deviceId.current,
            mock: true, // Set to false for real scraping
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        // Mark as completed before computing score
        hasCompletedRef.current = true;

        // Compute and save new impatience score
        const waitTime = Date.now() - startTimeRef.current;
        const newScore = computeImpatienceScore(
          rageClicksRef.current,
          totalClicksRef.current,
          tabSwitchesRef.current,
          waitTime
        );
        saveImpatienceScore(newScore);

        console.log('✅ Scrape complete!', {
          settings_used: result.settings_used,
          previous_impatience: previousImpatience.current,
          new_impatience: newScore,
          behavior: {
            rage_clicks: rageClicksRef.current,
            total_clicks: totalClicksRef.current,
            tab_switches: tabSwitchesRef.current,
            wait_time_ms: waitTime,
          },
        });

        // Navigate to results with the scrape data
        navigate('/results', { ...initialState, scrapeResult: result });
      } catch (err) {
        console.error('Scrape error:', err);
        setStatus('error');
        setErrorMsg(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    runScrape();
  }, [initialState, navigate]);

  if (!initialState) {
    return null;
  }

  if (status === 'error') {
    return (
      <div
        className="min-h-screen relative overflow-hidden flex items-center justify-center"
        style={{
          background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b3d 100%)',
        }}
      >
        <div className="text-center p-8">
          <h1 className="text-3xl text-red-400 mb-4">Error</h1>
          <p className="text-violet-300 mb-6">{errorMsg}</p>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-violet-600 text-white rounded-xl"
          >
            Go Back
          </button>
        </div>
      </div>
    );
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
