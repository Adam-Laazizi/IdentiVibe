import React, { useState } from 'react';

interface HistoryItem {
  query: string;
  timestamp: string;
  sources: {
    reddit?: string;
    youtube?: string;
    linkedin?: string;
    instagram?: string;
  };
  community_report: any;
}

const HISTORY_DATA: HistoryItem[] = [
  {
    query: "Game Theory community analysis",
    timestamp: "2025-01-17T14:30:00Z",
    sources: {
      reddit: "https://reddit.com/r/GameTheory",
      youtube: "https://youtube.com/@GameTheory"
    },
    community_report: {
      overall_archetype: "Theorizing Gamers",
      community_summary: "This community primarily consists of individuals deeply engaged with video game culture, particularly the analysis and theorization aspects.",
      user_dossiers: [
        {
          user_id: "UCTB-eLMDVxsRkzuMHavHwAg",
          identity_profile: {
            archetype: "Skeptical Observer",
            persona_description: "This user seems to offer terse, critical observations.",
            primary_interest: "Critique"
          }
        }
      ]
    }
  },
  {
    query: "Tech startup founders LinkedIn",
    timestamp: "2025-01-16T09:15:00Z",
    sources: {
      linkedin: "https://linkedin.com/in/sample",
      instagram: "https://instagram.com/techstartups"
    },
    community_report: {
      overall_archetype: "Tech Innovators",
      community_summary: "A group of early-stage founders sharing insights and networking.",
      user_dossiers: []
    }
  }
];

interface HistoryProps {
  navigate: (path: string) => void;
}

export default function History({ navigate }: HistoryProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const toggleExpanded = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">Search History</h1>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 text-sm text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Back to Search
          </button>
        </div>

        <div className="space-y-4">
          {HISTORY_DATA.map((item, index) => {
            const isExpanded = expandedIndex === index;

            return (
              <div
                key={index}
                className="border border-gray-200 rounded-lg p-5 bg-white"
              >
                <div className="mb-3">
                  <h2 className="text-lg font-medium text-gray-900 mb-1">
                    {item.query}
                  </h2>
                  <p className="text-sm text-gray-500">
                    {formatTimestamp(item.timestamp)}
                  </p>
                </div>

                {Object.keys(item.sources).length > 0 && (
                  <div className="mb-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Sources:</p>
                    <div className="flex flex-wrap gap-2">
                      {item.sources.reddit && (
                        <a
                          href={item.sources.reddit}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                        >
                          Reddit
                        </a>
                      )}
                      {item.sources.youtube && (
                        <a
                          href={item.sources.youtube}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                        >
                          YouTube
                        </a>
                      )}
                      {item.sources.linkedin && (
                        <a
                          href={item.sources.linkedin}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                        >
                          LinkedIn
                        </a>
                      )}
                      {item.sources.instagram && (
                        <a
                          href={item.sources.instagram}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                        >
                          Instagram
                        </a>
                      )}
                    </div>
                  </div>
                )}

                <button
                  onClick={() => toggleExpanded(index)}
                  className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  {isExpanded ? (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="18 15 12 9 6 15"></polyline>
                      </svg>
                      Hide JSON
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                      </svg>
                      View JSON
                    </>
                  )}
                </button>

                {isExpanded && (
                  <div className="mt-3 p-3 bg-gray-50 rounded border border-gray-200 overflow-x-auto">
                    <pre className="text-xs text-gray-700">
                      {JSON.stringify(item.community_report, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
