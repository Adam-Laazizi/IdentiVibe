import React, { useState } from 'react';
import { Search } from 'lucide-react';

export default function App() {
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = () => {
    if (searchQuery.trim()) {
      console.log('Searching for:', searchQuery);
      // Add your search logic here
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            identify.ai
          </h1>
          <p className="text-gray-600 text-lg">
            Discover and identify anything with AI
          </p>
        </div>

        <div className="relative">
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="What would you like to identify?"
              className="w-full px-6 py-5 pr-14 text-lg rounded-2xl border-2 border-gray-200 focus:border-blue-500 focus:outline-none shadow-lg transition-all duration-200 hover:shadow-xl"
            />
            <button
              onClick={handleSearch}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:shadow-lg transition-all duration-200 hover:scale-105"
            >
              <Search size={20} />
            </button>
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Try searching for images, text, or objects to identify
          </p>
        </div>
      </div>
    </div>
  );
}