import React, { useState, useEffect } from 'react';
import type { Sources, ResolveSourcesResponse } from '../types/sources';

type SourcesProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string, state?: ResolveSourcesResponse) => void;
};

export function Sources({ initialState, navigate }: SourcesProps) {
  const [sources, setSources] = useState<Sources | null>(null);
  const [query, setQuery] = useState('');

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

  const handleSubmit = () => {
    if (!sources) return;
    navigate('/results', { query, sources });
  };

  if (!sources) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
            Query: {query}
          </span>
        </div>

        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Edit social media sources
        </h1>

        <div className="space-y-4 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              YouTube
            </label>
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="#FF0000">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                </svg>
              </div>
              <input
                type="text"
                value={sources.youtubeUrl}
                onChange={(e) => updateSource('youtubeUrl', e.target.value)}
                placeholder="https://www.youtube.com/@..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
              />
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Instagram
            </label>
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="url(#instagram-gradient)">
                  <defs>
                    <linearGradient id="instagram-gradient" x1="0%" y1="100%" x2="100%" y2="0%">
                      <stop offset="0%" style={{ stopColor: '#FD5949', stopOpacity: 1 }} />
                      <stop offset="50%" style={{ stopColor: '#D6249F', stopOpacity: 1 }} />
                      <stop offset="100%" style={{ stopColor: '#285AEB', stopOpacity: 1 }} />
                    </linearGradient>
                  </defs>
                  <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z"/>
                </svg>
              </div>
              <input
                type="text"
                value={sources.instagramUrl}
                onChange={(e) => updateSource('instagramUrl', e.target.value)}
                placeholder="https://www.instagram.com/..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
              />
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              LinkedIn
            </label>
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="#0A66C2">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </div>
              <input
                type="text"
                value={sources.linkedinUrl}
                onChange={(e) => updateSource('linkedinUrl', e.target.value)}
                placeholder="https://www.linkedin.com/company/..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
              />
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Reddit
            </label>
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="#FF4500">
                  <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                </svg>
              </div>
              <input
                type="text"
                value={sources.redditUrl}
                onChange={(e) => updateSource('redditUrl', e.target.value)}
                placeholder="https://www.reddit.com/r/..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
              />
            </div>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
          >
            Back
          </button>
          <button
            onClick={handleSubmit}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}