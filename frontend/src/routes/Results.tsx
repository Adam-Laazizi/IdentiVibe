import { useEffect } from 'react';
import type { ResolveSourcesResponse } from '../types/sources';

type ResultsProps = {
  initialState: ResolveSourcesResponse | null;
  navigate: (path: string) => void;
};

export function Results({ initialState, navigate }: ResultsProps) {
  useEffect(() => {
    if (!initialState) {
      navigate('/');
    }
  }, [initialState, navigate]);

  if (!initialState) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Results</h1>

        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Query & Sources
          </h2>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-auto">
            <pre className="text-sm">{JSON.stringify(initialState, null, 2)}</pre>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              AI-generated identity
            </h2>
            <p className="text-gray-500 italic">Coming soon...</p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              AI image
            </h2>
            <p className="text-gray-500 italic">Coming soon...</p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Statistics
            </h2>
            <p className="text-gray-500 italic">Coming soon...</p>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
          >
            New Search
          </button>
        </div>
      </div>
    </div>
  );
}