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
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Reddit URL
            </label>
            <input
              type="text"
              value={sources.redditUrl}
              onChange={(e) => updateSource('redditUrl', e.target.value)}
              placeholder="https://www.reddit.com/r/..."
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
            />
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              YouTube URL
            </label>
            <input
              type="text"
              value={sources.youtubeUrl}
              onChange={(e) => updateSource('youtubeUrl', e.target.value)}
              placeholder="https://www.youtube.com/@..."
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
            />
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Instagram URL
            </label>
            <input
              type="text"
              value={sources.instagramUrl}
              onChange={(e) => updateSource('instagramUrl', e.target.value)}
              placeholder="https://www.instagram.com/..."
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
            />
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              LinkedIn URL
            </label>
            <input
              type="text"
              value={sources.linkedinUrl}
              onChange={(e) => updateSource('linkedinUrl', e.target.value)}
              placeholder="https://www.linkedin.com/company/..."
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm font-mono"
            />
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