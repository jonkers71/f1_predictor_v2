"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Calendar from "@/components/Calendar";

interface Prediction {
  rank: number;
  driver: string;
  team: string;
  time: string;
}

interface ActiveEvent {
  name: string;
  round: number;
  year: number;
  date: string;
  location: string;
}

export default function Dashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [event, setEvent] = useState<ActiveEvent | null>(null);
  const [syncProgress, setSyncProgress] = useState(0);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSynced, setLastSynced] = useState<string>("Never");

  useEffect(() => {
    // Initial data fetch
    fetch("http://localhost:8000/active_event")
      .then(res => {
        if (!res.ok) throw new Error(`Backend error: ${res.status}`);
        return res.json();
      })
      .then(data => {
        console.log("Fetched active event:", data);
        setEvent(data);
      })
      .catch(err => {
        console.error("Error fetching active event:", err);
        // Fallback to avoid empty state
        setEvent({
          name: "Data Loading...",
          round: 0,
          year: 2026,
          date: "",
          location: "Determining Location"
        });
      });

    // Mock predictions for demo
    setPredictions([
      { rank: 1, driver: "Max Verstappen", team: "Red Bull", time: "1:29.841" },
      { rank: 2, driver: "Lando Norris", team: "McLaren", time: "1:29.943" },
      { rank: 3, driver: "Charles Leclerc", team: "Ferrari", time: "1:30.102" },
      { rank: 4, driver: "Oscar Piastri", team: "McLaren", time: "1:30.150" },
      { rank: 5, driver: "George Russell", team: "Mercedes", time: "1:30.291" },
    ]);
  }, []);

  const handleSync = async () => {
    setIsSyncing(true);
    setSyncProgress(0);

    // Simulate steps for UI progress
    for (let i = 1; i <= 10; i++) {
      await new Promise(r => setTimeout(r, 200));
      setSyncProgress(i * 10);
    }

    try {
      const res = await fetch("http://localhost:8000/sync", { method: "POST" });
      const data = await res.json();
      const syncDate = new Date(data.last_synced);
      setLastSynced(!isNaN(syncDate.getTime()) ? syncDate.toLocaleTimeString() : "Just Now");
    } finally {
      setIsSyncing(false);
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
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <span className="w-2 h-8 bg-[#66fcf1] rounded-full"></span>
                {event?.round === 0 ? "PRE-SEASON PERFORMANCE PREDICTION" : "QUALIFYING PREDICTION"}
              </h2>
              {event && (
                <p className="text-xs text-[#66fcf1] mt-2 font-mono uppercase tracking-widest">
                  {event.year} • {event.round === 0 ? "TESTING SESSION" : `ROUND ${event.round}`} • {event.name}
                </p>
              )}
            </div>
            <span className="text-xs text-[#45a29e] font-mono">LIVE PREDICTION ENGINE V3.0</span>
          </div>

          <div className="space-y-3">
            {predictions.map((p) => (
              <div
                key={p.driver}
                className="group flex items-center justify-between p-4 bg-[#0b0c10] rounded-xl hover:border-[#66fcf1] border border-transparent transition-all cursor-pointer"
              >
                <div className="flex items-center gap-6">
                  <span className="text-3xl font-black text-[#45a29e] w-8">{p.rank}</span>
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
            ))}
          </div>
        </section>

        {/* Sidebar */}
        <aside className="space-y-8">
          <Calendar />

          <div className="bg-[#1f2833] rounded-2xl p-6 border border-[#45a29e]/20">
            <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest mb-4 flex justify-between">
              Session Insights
              {event && <span className="text-[10px] text-[#66fcf1]">{event.location}</span>}
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between text-sm">
                <span>Track Temp</span>
                <span className="text-[#66fcf1]">34.2°C</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Rain Risk</span>
                <span className="text-[#66fcf1]">12%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Track Evolution</span>
                <span className="text-[#66fcf1]">+0.4s</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
