import { useState } from 'react';

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
      <div className="absolute top-20 right-10 w-96 h-96 bg-violet-500 rounded-full opacity-20 blur-3xl animate-pulse"></div>
      <div className="absolute bottom-40 left-10 w-96 h-96 bg-cyan-500 rounded-full opacity-20 blur-3xl animate-pulse" style={{ animationDelay: '1.5s' }}></div>

      <div className="relative max-w-4xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-12">
          <div>
            <h1 className="text-4xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-300 to-cyan-400" style={{
              fontFamily: '"Bebas Neue", "Impact", sans-serif',
              textShadow: '0 0 30px rgba(139, 92, 246, 0.5)'
            }}>
              HISTORY ARCHIVE
            </h1>
            <p className="text-violet-300/70 mt-2" style={{
              fontFamily: '"Fira Code", "Courier New", monospace',
              fontSize: '0.875rem',
              letterSpacing: '0.1em'
            }}>
              &gt; Previous reconnaissance missions_
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="group px-6 py-3 text-violet-300 hover:text-violet-100 border border-violet-500/30 hover:border-violet-400/60 rounded-xl transition-all duration-300 backdrop-blur-sm hover:bg-violet-500/10"
            style={{ fontFamily: '"Fira Code", monospace' }}
          >
            <span className="inline-block transition-transform duration-300 group-hover:-translate-x-1">
              ← Return to Main
            </span>
          </button>
        </div>

        <div className="space-y-5">
          {HISTORY_DATA.map((item, index) => {
            const isExpanded = expandedIndex === index;

            return (
              <div
                key={index}
                className="relative group"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 to-cyan-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                <div className="relative bg-slate-900/80 backdrop-blur-sm border border-violet-500/30 rounded-2xl p-6 hover:border-violet-400/50 transition-all duration-300">
                  <div className="mb-4">
                    <h2 className="text-xl font-bold text-violet-100 mb-2" style={{
                      fontFamily: '"Orbitron", sans-serif'
                    }}>
                      {item.query}
                    </h2>
                    <p className="text-sm text-cyan-300/70" style={{
                      fontFamily: '"Fira Code", monospace'
                    }}>
                      TIMESTAMP: {formatTimestamp(item.timestamp)}
                    </p>
                  </div>

                  {Object.keys(item.sources).length > 0 && (
                    <div className="mb-5">
                      <p className="text-sm font-semibold text-violet-300 mb-3 tracking-wider" style={{
                        fontFamily: '"Fira Code", monospace',
                        textTransform: 'uppercase'
                      }}>
                        // Sources Located:
                      </p>
                      <div className="flex flex-wrap gap-3">
                        {item.sources.reddit && (
                          <a
                            href={item.sources.reddit}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 text-xs font-bold bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400/40 text-violet-100 rounded-lg hover:from-violet-500/60 hover:to-fuchsia-500/60 hover:border-violet-300/60 transition-all duration-300 backdrop-blur-sm"
                            style={{ fontFamily: '"Fira Code", monospace' }}
                          >
                            REDDIT
                          </a>
                        )}
                        {item.sources.youtube && (
                          <a
                            href={item.sources.youtube}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 text-xs font-bold bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400/40 text-violet-100 rounded-lg hover:from-violet-500/60 hover:to-fuchsia-500/60 hover:border-violet-300/60 transition-all duration-300 backdrop-blur-sm"
                            style={{ fontFamily: '"Fira Code", monospace' }}
                          >
                            YOUTUBE
                          </a>
                        )}
                        {item.sources.linkedin && (
                          <a
                            href={item.sources.linkedin}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 text-xs font-bold bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400/40 text-violet-100 rounded-lg hover:from-violet-500/60 hover:to-fuchsia-500/60 hover:border-violet-300/60 transition-all duration-300 backdrop-blur-sm"
                            style={{ fontFamily: '"Fira Code", monospace' }}
                          >
                            LINKEDIN
                          </a>
                        )}
                        {item.sources.instagram && (
                          <a
                            href={item.sources.instagram}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 text-xs font-bold bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400/40 text-violet-100 rounded-lg hover:from-violet-500/60 hover:to-fuchsia-500/60 hover:border-violet-300/60 transition-all duration-300 backdrop-blur-sm"
                            style={{ fontFamily: '"Fira Code", monospace' }}
                          >
                            INSTAGRAM
                          </a>
                        )}
                      </div>
                    </div>
                  )}

                  <button
                    onClick={() => toggleExpanded(index)}
                    className="flex items-center gap-2 text-sm text-cyan-300 hover:text-cyan-100 transition-colors group/btn"
                    style={{ fontFamily: '"Fira Code", monospace' }}
                  >
                    {isExpanded ? (
                      <>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform duration-300 group-hover/btn:-translate-y-0.5">
                          <polyline points="18 15 12 9 6 15"></polyline>
                        </svg>
                        [HIDE_DATA]
                      </>
                    ) : (
                      <>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform duration-300 group-hover/btn:translate-y-0.5">
                          <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                        [VIEW_DATA]
                      </>
                    )}
                  </button>

                  {isExpanded && (
                    <div className="mt-4 p-4 bg-slate-950/60 backdrop-blur-sm rounded-xl border border-cyan-500/30 overflow-x-auto">
                      <pre className="text-xs text-cyan-100/90" style={{
                        fontFamily: '"Fira Code", "Courier New", monospace'
                      }}>
                        {JSON.stringify(item.community_report, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
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