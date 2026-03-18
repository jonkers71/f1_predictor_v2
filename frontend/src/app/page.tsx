"use client";

import { useState, useEffect } from "react";
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
  weights: Weights;
}

interface Prediction {
  rank: number;
  driver: string;
  team: string;
  time: string;
  breakdown: Breakdown;
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

export default function Dashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [event, setEvent] = useState<ActiveEvent | null>(null);
  const [sessionType, setSessionType] = useState<"Q" | "R">("Q");
  const [syncProgress, setSyncProgress] = useState(0);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSynced, setLastSynced] = useState<string>("Never");
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<string>("");
  const [predictionError, setPredictionError] = useState<string>("");
  const [dataSources, setDataSources] = useState<DataSources | null>(null);
  const [modelStatus, setModelStatus] = useState<{status: string, last_trained: string, feature_version: string} | null>(null);

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

  useEffect(() => {
    // Restore sync timestamp from backend
    fetch("http://localhost:8000/sync/status")
      .then(res => res.json())
      .then(data => {
        if (data.last_synced) {
          const d = new Date(data.last_synced);
          setLastSynced(!isNaN(d.getTime()) ? d.toLocaleTimeString() : "Never");
        }
      })
    // Model Status
    fetch("http://localhost:8000/model/status")
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(() => {});

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
  }, [sessionType]);

  const handleSync = async () => {
    setIsSyncing(true);
    setSyncProgress(0);

    for (let i = 1; i <= 10; i++) {
        await new Promise(r => setTimeout(r, 100));
        setSyncProgress(i * 10);
    }

    try {
      const res = await fetch("http://localhost:8000/sync", { method: "POST" });
      const data = await res.json();
      const syncDate = new Date(data.last_synced);
      setLastSynced(!isNaN(syncDate.getTime()) ? syncDate.toLocaleTimeString() : "Just Now");
      fetchPredictions(sessionType);
    } finally {
      setIsSyncing(false);
    }
  };

  const sourceTypeLabel = (type: string) => {
    switch (type) {
      case "live_practice": return "Live Practice Data";
      case "circuit_history": return "Circuit History";
      case "none": return "No Data";
      default: return type;
    }
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
              className={`${isSyncing ? 'bg-gray-600' : 'bg-[#45a29e] hover:bg-[#66fcf1]'} text-[#0b0c10] px-6 py-2 rounded-full font-bold transition-all uppercase text-xs`}
            >
              {isSyncing ? 'Syncing...' : 'Sync Latest Data'}
            </button>
          </div>
          <p className="text-[10px] text-[#45a29e] uppercase font-mono">Last Synced: {lastSynced}</p>
        </div>
      </header>

      {/* Progress Bar */}
      {isSyncing && (
        <div className="w-full h-1 bg-gray-800 mb-8 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#66fcf1] transition-all duration-300"
            style={{ width: `${syncProgress}%` }}
          ></div>
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
                      <div className="text-lg font-bold text-white group-hover:text-[#66fcf1] transition-colors">
                        {p.driver.toUpperCase()}
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
              {/* Data Source Type */}
              <div className="flex justify-between text-sm">
                <span>Source</span>
                <span className={`font-mono ${dataSources?.type === 'live_practice' ? 'text-green-400' : 'text-[#66fcf1]'}`}>
                  {dataSources ? sourceTypeLabel(dataSources.type) : '—'}
                </span>
              </div>

              {/* Circuit */}
              <div className="flex justify-between text-sm">
                <span>Circuit</span>
                <span className="text-[#66fcf1] font-mono">{dataSources?.circuit || '—'}</span>
              </div>

              {/* Long Stint Data */}
              {sessionType === "R" && (
                <div className="flex justify-between text-sm">
                  <span>Long Stint Data</span>
                  <span className={dataSources?.has_long_stint_data ? 'text-green-400' : 'text-yellow-500'}>
                    {dataSources?.has_long_stint_data ? '✓ Available' : '✗ Not Available'}
                  </span>
                </div>
              )}

              {/* Season Adjustment */}
              <div className="flex justify-between text-sm">
                <span>Season Form</span>
                <span className="text-green-400">
                  {dataSources?.current_season_adjustment ? '✓ Applied' : '—'}
                </span>
              </div>

              {/* Sessions Used */}
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

              {/* Notes */}
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
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
