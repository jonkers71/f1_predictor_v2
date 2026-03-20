"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import Calendar from "@/components/Calendar";
import WeightingBreakdown from "@/components/WeightingBreakdown";

interface Weights {
  base: number;
  slope: number;
  consistency: number;
  conversion: number;
  team: number;
}

interface Breakdown {
  base_score: number;
  slope_score: number;
  consistency_score: number;
  sunday_conversion: number;
  team_score: number;
  final_score: number;
  confidence_score: number;
  confidence_reasons: string[];
  weights: Weights;
}

interface Prediction {
  rank: number;
  driver: string;
  team: string;
  time: string;
  breakdown: Breakdown;
  confidence: number;
}

interface ActiveEvent {
  name: string;
  round: number;
  year: number;
  date: string;
  location: string;
}

interface DataSources {
  type: string;
  circuit: string;
  sessions_used: string[];
  current_season_adjustment: boolean;
  has_long_stint_data: boolean;
  notes: string[];
}

interface HealthLayer {
  status: "ready" | "partial" | "missing";
  detail: string;
}

interface DataHealth {
  cache_exists: boolean;
  total_sessions_cached: number;
  layers: {
    rookie_data: HealthLayer;
    sunday_conversion: HealthLayer;
    circuit_history: HealthLayer;
    rolling_ratings: HealthLayer;
  };
}

interface ModelStatus {
  status: string;
  last_trained: string;
  feature_version: string;
  training_type: string;
  sessions_trained_on: number;
}

export default function Dashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [event, setEvent] = useState<ActiveEvent | null>(null);
  const [sessionType, setSessionType] = useState<"Q" | "R">("Q");
  const [syncProgress, setSyncProgress] = useState(0);
  const [syncStage, setSyncStage] = useState("Idle");
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSynced, setLastSynced] = useState<string>("Never");
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<string>("");
  const [predictionError, setPredictionError] = useState<string>("");
  const [dataSources, setDataSources] = useState<DataSources | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [dataHealth, setDataHealth] = useState<DataHealth | null>(null);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainProgress, setRetrainProgress] = useState(0);
  const [retrainStage, setRetrainStage] = useState("Idle");
  const syncPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retrainPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchPredictions = (type: "Q" | "R") => {
    setPredictionError("");
    fetch(`http://localhost:8000/predict/current?session_type=${type}`)
      .then(res => {
        if (!res.ok) throw new Error(`Backend error: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setPredictions(data.predictions || []);
        setBaseline(data.baseline || "");
        setDataSources(data.data_sources || null);
        if (!data.predictions || data.predictions.length === 0) {
          setPredictionError(`No predictions available (Baseline: ${data.baseline || 'None'})`);
        }
      })
      .catch(err => {
        console.error("Error fetching predictions:", err);
        setPredictionError("Could not load predictions. Is the backend running?");
      });
  };

  const fetchDataHealth = () => {
    fetch("http://localhost:8000/data/health")
      .then(res => res.json())
      .then(data => setDataHealth(data))
      .catch(() => {});
  };

  const fetchModelStatus = () => {
    fetch("http://localhost:8000/model/status")
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(err => console.error("Model status fetch failed:", err));
  };

  useEffect(() => {
    // Restore sync timestamp
    fetch("http://localhost:8000/sync/status")
      .then(res => res.json())
      .then(data => {
        if (data.last_synced) {
          const d = new Date(data.last_synced);
          setLastSynced(!isNaN(d.getTime()) ? d.toLocaleString() : "Never");
        }
      })
      .catch(() => {});

    // Model Status
    setTimeout(() => fetchModelStatus(), 500);

    // Data Health
    setTimeout(() => fetchDataHealth(), 800);

    // Initial data fetch
    fetch("http://localhost:8000/active_event")
      .then(res => {
        if (!res.ok) throw new Error(`Backend error: ${res.status}`);
        return res.json();
      })
      .then(data => {
        setEvent(data);
        fetchPredictions(sessionType);
      })
      .catch(err => {
        console.error("Error fetching active event:", err);
        setEvent({
          name: "Data Loading...",
          round: 0,
          year: 2026,
          date: "",
          location: "Determining Location"
        });
      });

    return () => {
      if (syncPollRef.current) clearInterval(syncPollRef.current);
      if (retrainPollRef.current) clearInterval(retrainPollRef.current);
    };
  }, [sessionType]);

  // ── Real Sync with live progress polling ──
  const handleSync = async () => {
    if (isSyncing) return;
    setIsSyncing(true);
    setSyncProgress(0);
    setSyncStage("Starting...");

    try {
      await fetch("http://localhost:8000/sync", { method: "POST" });
    } catch {
      setIsSyncing(false);
      return;
    }

    // Poll /sync/status every second
    syncPollRef.current = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/sync/status");
        const data = await res.json();
        setSyncProgress(data.progress ?? 0);
        setSyncStage(data.stage ?? "");
        if (!data.is_syncing) {
          clearInterval(syncPollRef.current!);
          setIsSyncing(false);
          if (data.last_synced) {
            const d = new Date(data.last_synced);
            setLastSynced(!isNaN(d.getTime()) ? d.toLocaleString() : "Just Now");
          }
          fetchDataHealth();
          fetchPredictions(sessionType);
        }
      } catch {
        clearInterval(syncPollRef.current!);
        setIsSyncing(false);
      }
    }, 1000);
  };

  // ── Retrain with live progress polling ──
  const handleRetrain = async () => {
    if (isRetraining) return;
    setIsRetraining(true);
    setRetrainProgress(0);
    setRetrainStage("Starting...");

    try {
      await fetch("http://localhost:8000/model/retrain", { method: "POST" });
    } catch {
      setIsRetraining(false);
      return;
    }

    retrainPollRef.current = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/model/retrain/progress");
        const data = await res.json();
        setRetrainProgress(data.progress ?? 0);
        setRetrainStage(data.stage ?? "");
        if (!data.is_training) {
          clearInterval(retrainPollRef.current!);
          setIsRetraining(false);
          fetchModelStatus();
          fetchPredictions(sessionType);
        }
      } catch {
        clearInterval(retrainPollRef.current!);
        setIsRetraining(false);
      }
    }, 1000);
  };

  const sourceTypeLabel = (type: string) => {
    switch (type) {
      case "live_practice": return "Live Practice Data";
      case "circuit_history": return "Circuit History";
      case "none": return "No Data";
      default: return type;
    }
  };

  const healthColor = (status: string) => {
    if (status === "ready") return "text-green-400";
    if (status === "partial") return "text-yellow-400";
    return "text-red-400";
  };

  const healthDot = (status: string) => {
    if (status === "ready") return "bg-green-400";
    if (status === "partial") return "bg-yellow-400";
    return "bg-red-500";
  };

  const confidenceColor = (score: number) => {
    if (score >= 80) return "bg-green-500";
    if (score >= 50) return "bg-yellow-400";
    return "bg-red-500";
  };

  return (
    <main className="min-h-screen bg-[#0b0c10] text-[#c5c6c7] p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-12 border-b border-[#45a29e] pb-6">
        <div>
          <h1 className="text-4xl font-bold text-[#66fcf1] tracking-tighter uppercase italic">
            F1 Predictor <span className="text-white">v2</span>
          </h1>
          <p className="text-sm text-[#45a29e] mt-1 tracking-widest uppercase">
            Quantified Performance Analytics
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex gap-4">
            <Link
              href="/backtest"
              className="border border-[#45a29e] text-[#45a29e] hover:text-[#66fcf1] hover:border-[#66fcf1] px-6 py-2 rounded-full font-bold transition-all uppercase text-xs"
            >
              Backtesting
            </Link>
            <button
              onClick={handleSync}
              disabled={isSyncing}
              className={`${isSyncing ? 'bg-gray-600 cursor-not-allowed' : 'bg-[#45a29e] hover:bg-[#66fcf1]'} text-[#0b0c10] px-6 py-2 rounded-full font-bold transition-all uppercase text-xs`}
            >
              {isSyncing ? `Syncing... ${syncProgress}%` : 'Sync Latest Data'}
            </button>
          </div>
          <p className="text-[10px] text-[#45a29e] uppercase font-mono">Last Synced: {lastSynced}</p>
        </div>
      </header>

      {/* Sync Progress Bar */}
      {isSyncing && (
        <div className="mb-6">
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#66fcf1] transition-all duration-500"
              style={{ width: `${syncProgress}%` }}
            />
          </div>
          <p className="text-[10px] text-[#45a29e] font-mono mt-1 uppercase tracking-widest">{syncStage}</p>
        </div>
      )}

      {/* Retrain Progress Bar */}
      {isRetraining && (
        <div className="mb-6">
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-400 transition-all duration-500"
              style={{ width: `${retrainProgress}%` }}
            />
          </div>
          <p className="text-[10px] text-purple-400 font-mono mt-1 uppercase tracking-widest">{retrainStage}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Prediction Panel */}
        <section className="lg:col-span-2 bg-[#1f2833] rounded-2xl p-6 shadow-2xl border border-[#45a29e]/20">
          <div className="flex justify-between items-end mb-8">
            <div>
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setSessionType("Q")}
                  className={`text-2xl font-bold flex items-center gap-2 transition-all ${sessionType === "Q" ? 'text-white' : 'text-white/30 hover:text-white/60'}`}
                >
                  <span className={`w-2 h-8 rounded-full ${sessionType === "Q" ? 'bg-[#66fcf1]' : 'bg-white/10'}`}></span>
                  QUALIFYING
                </button>
                <button
                  onClick={() => setSessionType("R")}
                  className={`text-2xl font-bold flex items-center gap-2 transition-all ${sessionType === "R" ? 'text-white' : 'text-white/30 hover:text-white/60'}`}
                >
                  <span className={`w-2 h-8 rounded-full ${sessionType === "R" ? 'bg-[#ff5252]' : 'bg-white/10'}`}></span>
                  RACE PREDICTION
                </button>
              </div>
              {event && (
                <p className="text-xs text-[#66fcf1] mt-2 font-mono uppercase tracking-widest">
                  {event.year} • ROUND {event.round} • {event.name}
                </p>
              )}
            </div>
            <span className="text-xs text-[#45a29e] font-mono">LIVE PREDICTION ENGINE V4.0</span>
          </div>

          <div className="space-y-3">
            {predictions.map((p) => (
              <div key={p.driver}>
                <div
                  onClick={() => setSelectedDriver(selectedDriver === p.driver ? null : p.driver)}
                  className={`group flex items-center justify-between p-4 bg-[#0b0c10] rounded-xl border transition-all cursor-pointer ${selectedDriver === p.driver ? 'border-[#66fcf1] shadow-[0_0_15px_rgba(102,252,241,0.1)]' : 'border-transparent hover:border-[#66fcf1]/30'}`}
                >
                  <div className="flex items-center gap-6">
                    <span className={`text-3xl font-black w-8 ${selectedDriver === p.driver ? 'text-[#66fcf1]' : 'text-[#45a29e]'}`}>{p.rank}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <div className="text-lg font-bold text-white group-hover:text-[#66fcf1] transition-colors">
                          {p.driver.toUpperCase()}
                        </div>
                        {/* Confidence dot */}
                        {p.breakdown?.confidence_score !== undefined && (
                          <div className="relative group/conf">
                            <div className={`w-2 h-2 rounded-full ${confidenceColor(p.breakdown.confidence_score)}`} />
                            <div className="absolute left-4 top-0 z-10 hidden group-hover/conf:block bg-[#1f2833] border border-[#45a29e]/40 rounded-lg p-2 text-[10px] w-40 shadow-xl">
                              <p className="text-[#66fcf1] font-bold mb-1">Confidence: {p.breakdown.confidence_score}%</p>
                              {p.breakdown.confidence_reasons?.map((r, i) => (
                                <p key={i} className="text-[#c5c6c7]/70">• {r}</p>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="text-xs text-[#45a29e] uppercase tracking-tighter">
                        {p.team}
                      </div>
                    </div>
                  </div>
                  <div className="text-xl font-mono text-[#66fcf1]">
                    {p.time}
                  </div>
                </div>
                {selectedDriver === p.driver && (
                  <WeightingBreakdown breakdown={p.breakdown} driverName={p.driver} />
                )}
              </div>
            ))}
          </div>
          {predictionError && predictions.length === 0 && (
            <div className="mt-6 p-6 text-center text-[#45a29e] border border-[#45a29e]/20 rounded-xl">
              <p className="text-sm italic">{predictionError}</p>
              <p className="text-[10px] mt-2 uppercase tracking-widest opacity-50">Check backend logs for details</p>
            </div>
          )}
        </section>

        {/* Sidebar */}
        <aside className="space-y-8">
          <Calendar />

          {/* Data Sources & Session Insights */}
          <div className="bg-[#1f2833] rounded-2xl p-6 border border-[#45a29e]/20">
            <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest mb-4 flex justify-between">
              Prediction Data
              {event && <span className="text-[10px] text-[#66fcf1]">{event.location}</span>}
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span>Source</span>
                <span className={`font-mono ${dataSources?.type === 'live_practice' ? 'text-green-400' : 'text-[#66fcf1]'}`}>
                  {dataSources ? sourceTypeLabel(dataSources.type) : '—'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Circuit</span>
                <span className="text-[#66fcf1] font-mono">{dataSources?.circuit || '—'}</span>
              </div>
              {sessionType === "R" && (
                <div className="flex justify-between text-sm">
                  <span>Long Stint Data</span>
                  <span className={dataSources?.has_long_stint_data ? 'text-green-400' : 'text-yellow-400'}>
                    {dataSources?.has_long_stint_data ? '✓ Available' : '⚠ Unavailable'}
                  </span>
                </div>
              )}
              <div className="flex justify-between text-sm">
                <span>Season Form</span>
                <span className="text-green-400">
                  {dataSources?.current_season_adjustment ? '✓ Applied' : '—'}
                </span>
              </div>
              {dataSources?.sessions_used && dataSources.sessions_used.length > 0 && (
                <div className="pt-3 border-t border-[#45a29e]/20">
                  <div className="text-[10px] text-[#45a29e] uppercase font-bold mb-2">Sessions Used</div>
                  <div className="flex flex-wrap gap-1">
                    {dataSources.sessions_used.map((s, i) => (
                      <span key={i} className="text-[10px] bg-[#0b0c10] px-2 py-1 rounded text-[#66fcf1] font-mono border border-[#45a29e]/20">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {dataSources?.notes && dataSources.notes.length > 0 && (
                <div className="pt-3 border-t border-[#45a29e]/20">
                  <div className="text-[10px] text-[#45a29e] uppercase font-bold mb-2">Notes</div>
                  {dataSources.notes.map((note, i) => (
                    <p key={i} className="text-[10px] text-[#c5c6c7]/70 italic mb-1">{note}</p>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Data Health Panel */}
          <div className="bg-[#1f2833] rounded-2xl p-6 border border-[#45a29e]/20">
            <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest mb-4 flex justify-between items-center">
              Data Health
              <span className="text-[10px] text-[#66fcf1] font-mono">
                {dataHealth ? `${dataHealth.total_sessions_cached} sessions cached` : 'Loading...'}
              </span>
            </h3>
            {dataHealth ? (
              <div className="space-y-3">
                {Object.entries(dataHealth.layers).map(([key, layer]) => (
                  <div key={key} className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 mt-0.5 ${healthDot(layer.status)}`} />
                      <span className="text-xs capitalize">{key.replace(/_/g, ' ')}</span>
                    </div>
                    <span className={`text-[10px] font-mono text-right ${healthColor(layer.status)}`}>
                      {layer.status.toUpperCase()}
                    </span>
                  </div>
                ))}
                {!dataHealth.cache_exists && (
                  <p className="text-[10px] text-red-400 italic mt-2">Cache directory not found. Run a sync to create it.</p>
                )}
              </div>
            ) : (
              <p className="text-xs text-[#45a29e]/50 italic">Checking cache...</p>
            )}
          </div>

          {/* Model Health Status */}
          <div className="bg-[#1f2833] rounded-2xl p-6 border border-[#45a29e]/20">
            <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest mb-4">
              XGBoost Agent Status
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span>Model State</span>
                <span className={`font-mono font-bold ${modelStatus?.status === 'active' ? 'text-green-400' : 'text-red-400'}`}>
                  {modelStatus?.status?.toUpperCase() || 'OFFLINE'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Last Trained</span>
                <span className="text-[#66fcf1] font-mono text-xs">
                  {modelStatus && modelStatus.last_trained !== 'never'
                    ? new Date(modelStatus.last_trained).toLocaleString()
                    : 'System Default'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Feature Set</span>
                <span className="text-[#45a29e] font-mono text-[10px]">{modelStatus?.feature_version || 'v1.0'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Training Type</span>
                <span className={`font-mono text-[10px] ${modelStatus?.training_type === 'real' ? 'text-green-400' : 'text-yellow-400'}`}>
                  {modelStatus?.training_type === 'real'
                    ? `Real Data (${modelStatus.sessions_trained_on} sessions)`
                    : 'Synthetic'}
                </span>
              </div>
              {/* Retrain Button */}
              <div className="pt-3 border-t border-[#45a29e]/20">
                <button
                  onClick={handleRetrain}
                  disabled={isRetraining}
                  className={`w-full py-2 rounded-lg text-xs font-bold uppercase tracking-widest transition-all ${
                    isRetraining
                      ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                      : 'bg-purple-900/50 hover:bg-purple-700/60 text-purple-300 border border-purple-700/40'
                  }`}
                >
                  {isRetraining ? `Retraining... ${retrainProgress}%` : 'Retrain Model'}
                </button>
                {isRetraining && (
                  <p className="text-[10px] text-purple-400 font-mono mt-1 text-center">{retrainStage}</p>
                )}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
