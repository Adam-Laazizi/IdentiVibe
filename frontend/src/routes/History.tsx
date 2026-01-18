import { useEffect, useState } from 'react';
import { getHistory, getSearchById } from '../lib/mongodbApi';

interface HistoryRow {
  _id: string;
  query: string;
  platforms: any[];
  createdAt: string;
}

interface HistoryApiResponse {
  success: boolean;
  count: number;
  data: HistoryRow[];
}

interface SearchByIdResponse {
  success: boolean;
  data: any;
}

interface HistoryProps {
  navigate: (path: string) => void;
}

export default function History({ navigate }: HistoryProps) {
  const [items, setItems] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detailsById, setDetailsById] = useState<Record<string, any>>({});
  const [detailsLoadingId, setDetailsLoadingId] = useState<string | null>(null);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError('');
      try {
        const res: HistoryApiResponse = await getHistory('user123', 20);
        setItems(res?.data || []);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to fetch history');
      } finally {
        setLoading(false);
      }
    };

    run();
  }, []);

  const toggleExpanded = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }

    setExpandedId(id);

    if (detailsById[id]) return;

    setDetailsLoadingId(id);
    setError('');

    try {
      const res: SearchByIdResponse = await getSearchById(id);
      setDetailsById((prev) => ({ ...prev, [id]: res?.data }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch search details');
    } finally {
      setDetailsLoadingId(null);
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

        {error && (
          <div className="mb-6 relative overflow-hidden p-5 bg-red-950/50 backdrop-blur-sm border border-red-500/50 rounded-xl animate-pulse">
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/10 to-transparent"></div>
            <p className="relative text-red-300 font-medium" style={{ fontFamily: '"Fira Code", monospace' }}>
              ⚠ ERROR: {error}
            </p>
          </div>
        )}

        {loading ? (
          <div className="text-violet-300" style={{ fontFamily: '"Fira Code", monospace' }}>
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full animate-spin"></div>
              Loading history...
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {items.length === 0 ? (
              <div className="text-violet-300/70" style={{ fontFamily: '"Fira Code", monospace' }}>
                No history found for user123.
              </div>
            ) : (
              items.map((item) => {
                const isExpanded = expandedId === item._id;
                const detail = detailsById[item._id];

                return (
                  <div
                    key={item._id}
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
                          TIMESTAMP: {formatTimestamp(item.createdAt)}
                        </p>
                      </div>

                      {Array.isArray(item.platforms) && item.platforms.length > 0 && (
                        <div className="mb-5">
                          <p className="text-sm font-semibold text-violet-300 mb-3 tracking-wider" style={{
                            fontFamily: '"Fira Code", monospace',
                            textTransform: 'uppercase'
                          }}>
                            // Platforms:
                          </p>
                          <div className="flex flex-wrap gap-3">
                            {item.platforms.map((p, idx) => (
                              <span
                                key={idx}
                                className="px-4 py-2 text-xs font-bold bg-gradient-to-r from-violet-600/50 to-fuchsia-600/50 border border-violet-400/40 text-violet-100 rounded-lg backdrop-blur-sm"
                                style={{ fontFamily: '"Fira Code", monospace' }}
                              >
                                {typeof p === 'string' ? p : JSON.stringify(p)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      <button
                        onClick={() => toggleExpanded(item._id)}
                        className="flex items-center gap-2 text-sm text-cyan-300 hover:text-cyan-100 transition-colors group/btn"
                        style={{ fontFamily: '"Fira Code", monospace' }}
                      >
                        {isExpanded ? (
                          <>
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              className="transition-transform duration-300 group-hover/btn:-translate-y-0.5"
                            >
                              <polyline points="18 15 12 9 6 15"></polyline>
                            </svg>
                            [HIDE_DATA]
                          </>
                        ) : (
                          <>
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              className="transition-transform duration-300 group-hover/btn:translate-y-0.5"
                            >
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
                            {detailsLoadingId === item._id
                              ? 'Loading...'
                              : JSON.stringify(
                                  detail?.geminiResult ?? detail ?? {},
                                  null,
                                  2
                                )}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
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