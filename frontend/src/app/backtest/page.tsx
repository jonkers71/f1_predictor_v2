"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface ComparisonRow {
    driver: string;
    team: string;
    actual_rank: number;
    predicted_rank: number;
    actual_time: string;
    predicted_time: string;
    delta_error: number;
}

interface SessionInfo {
    name: string;
    code: string;
}

interface EventInfo {
    round: number;
    name: string;
    is_sprint: boolean;
    sessions: SessionInfo[];
}

export default function BacktestPage() {
    const [year, setYear] = useState(2024);
    const [round, setRound] = useState(1);
    const [sessionType, setSessionType] = useState("Q");
    const [schedule, setSchedule] = useState<EventInfo[]>([]);
    const [availableSessions, setAvailableSessions] = useState<SessionInfo[]>([]);

    const [results, setResults] = useState<ComparisonRow[]>([]);
    const [mae, setMae] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [eventInfo, setEventInfo] = useState<string>("");

    // Fetch schedule when year changes
    const fetchSchedule = async (selectedYear: number) => {
        try {
            const res = await fetch(`http://localhost:8000/schedule/${selectedYear}`);
            if (res.ok) {
                const data = await res.json();
                setSchedule(data);

                // Set default round to 1 if schedule exists
                if (data.length > 0) {
                    const firstEvent = data[0];
                    setRound(firstEvent.round);
                    setAvailableSessions(firstEvent.sessions);
                    // Find a sensible default session (Q or the last one)
                    const defSession = firstEvent.sessions.find((s: any) => s.code === 'Q') || firstEvent.sessions[firstEvent.sessions.length - 1];
                    setSessionType(defSession?.code || "Q");
                } else {
                    setRound(1);
                    setAvailableSessions([]);
                    setSessionType("Q");
                }
            } else {
                setSchedule([]);
                setRound(1);
                setAvailableSessions([]);
                setSessionType("Q");
            }
        } catch (err) {
            console.error("Failed to fetch schedule", err);
            setSchedule([]);
            setRound(1);
            setAvailableSessions([]);
            setSessionType("Q");
        }
    };

    // Initial load and re-fetch when year changes
    useEffect(() => {
        fetchSchedule(year);
    }, [year]);

    // Handle round change to update available sessions
    const handleRoundChange = (roundNum: number) => {
        setRound(roundNum);
        const event = schedule.find(e => e.round === roundNum);
        if (event) {
            setAvailableSessions(event.sessions);
            // If current sessionType isn't in new sessions, reset it
            if (!event.sessions.find(s => s.code === sessionType)) {
                const defSession = event.sessions.find(s => s.code === 'Q') || event.sessions[event.sessions.length - 1];
                setSessionType(defSession?.code || "Q");
            }
        } else {
            setAvailableSessions([]);
            setSessionType("Q");
        }
    };

    const runBacktest = async () => {
        setLoading(true);
        setProgress(0);
        setError(null);
        setResults([]);
        setMae(null);
        setStatus("Initializing Request...");

        // Simulate multi-stage progress
        const stages = [
            "Contacting F1 Servers...",
            "Downloading Telemetry Data...",
            "Processing Lap Times...",
            "Running XGBoost Ranker...",
            "Calculating Accuracy Delta..."
        ];

        let stageIdx = 0;
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev < 90) {
                    if (prev % 20 === 0 && stageIdx < stages.length) {
                        setStatus(stages[stageIdx++]);
                    }
                    return prev + 2;
                }
                return prev;
            });
        }, 500);

        try {
            const res = await fetch(`http://localhost:8000/backtest/${year}/${round}/${sessionType}`);
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || `Server Error: ${res.status}`);
            }
            const data = await res.json();
            clearInterval(interval);
            setProgress(100);
            setStatus("Analysis Complete");
            setResults(data.results || []);
            setMae(typeof data.mean_absolute_error === 'number' ? data.mean_absolute_error : null);
            setEventInfo(data.event || "");
        } catch (err: any) {
            console.error(err);
            setError(err.message || "Failed to reach backend");
            clearInterval(interval);
        } finally {
            setTimeout(() => {
                setLoading(false);
                if (!error) setProgress(0);
            }, 800);
        }
    };

    return (
        <main className="min-h-screen bg-[#0b0c10] text-[#c5c6c7] p-8">
            <header className="flex flex-col md:flex-row justify-between items-center mb-12 border-b border-[#45a29e] pb-6 gap-6">
                <div>
                    <Link href="/" className="text-[#66fcf1] text-xs uppercase tracking-widest hover:underline mb-2 block">← Back to Dashboard</Link>
                    <h1 className="text-4xl font-bold text-white tracking-tighter uppercase italic">
                        Accuracy <span className="text-[#66fcf1]">Analysis</span>
                    </h1>
                </div>

                <div className="flex flex-wrap gap-4 items-end justify-center">
                    <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-[#45a29e] uppercase font-bold">Year</label>
                        <select
                            value={year}
                            onChange={(e) => {
                                const y = parseInt(e.target.value);
                                setYear(y);
                                // fetchSchedule(y); // useEffect will handle this
                            }}
                            className="bg-[#1f2833] border border-[#45a29e]/30 px-3 py-2 rounded text-white text-sm"
                        >
                            <option value={2023}>2023</option>
                            <option value={2024}>2024</option>
                            <option value={2025}>2025</option>
                            <option value={2026}>2026</option>
                        </select>
                    </div>

                    <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-[#45a29e] uppercase font-bold">Round</label>
                        <select
                            value={round}
                            onChange={(e) => handleRoundChange(parseInt(e.target.value))}
                            className="bg-[#1f2833] border border-[#45a29e]/30 px-3 py-2 rounded text-white text-sm min-w-[200px]"
                        >
                            {schedule.map(event => (
                                <option key={event.round} value={event.round}>
                                    R{event.round}: {event.name} {event.is_sprint ? '🔴 S' : ''}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-[#45a29e] uppercase font-bold">Session</label>
                        <select
                            value={sessionType}
                            onChange={(e) => setSessionType(e.target.value)}
                            className="bg-[#1f2833] border border-[#45a29e]/30 px-3 py-2 rounded text-white text-sm"
                        >
                            {availableSessions.map(s => (
                                <option key={s.code} value={s.code}>{s.name}</option>
                            ))}
                        </select>
                    </div>

                    <button
                        onClick={runBacktest}
                        disabled={loading || schedule.length === 0}
                        className="bg-[#66fcf1] text-[#0b0c10] px-6 py-2 rounded font-bold hover:brightness-110 transition-all uppercase text-xs h-[38px] disabled:opacity-50"
                    >
                        {loading ? "Analyzing..." : "Run Analysis"}
                    </button>
                </div>
            </header>

            {/* Progress / Status Bar */}
            {loading && (
                <div className="mb-12">
                    <div className="flex justify-between text-[10px] text-[#66fcf1] uppercase font-mono mb-2 tracking-widest">
                        <span>{status}</span>
                        <span>{progress}%</span>
                    </div>
                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden border border-white/5">
                        <div
                            className="h-full bg-gradient-to-r from-[#45a29e] to-[#66fcf1] transition-all duration-300 shadow-[0_0_10px_#66fcf1]"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="mb-8 p-6 bg-red-900/20 border-l-4 border-red-500 rounded-r-xl">
                    <h4 className="text-red-500 font-bold uppercase text-xs mb-1">Backtest Error</h4>
                    <p className="text-sm text-red-200">{error}</p>
                    <button
                        onClick={() => setError(null)}
                        className="mt-4 text-[10px] uppercase text-red-500 hover:text-white underline"
                    >
                        Dismiss
                    </button>
                </div>
            )}

            <div className="flex justify-between items-end mb-6">
                <div className="space-y-1">
                    <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest">
                        Analysis Target: <span className="text-white">{sessionType === 'Q' ? 'Qualifying' : sessionType === 'R' ? 'Grand Prix' : sessionType === 'SQ' ? 'Sprint Qualifying' : 'Sprint Race'}</span>
                    </h3>
                    {eventInfo && (
                        <p className="text-xs text-[#66fcf1] font-mono">
                            {year} • {eventInfo.toUpperCase()}
                        </p>
                    )}
                </div>
            </div>

            {typeof mae === 'number' && (
                <div className="mb-8 grid grid-cols-1 md:grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2">
                    <div className="bg-[#1f2833] border-l-4 border-[#66fcf1] p-4 rounded-r-xl shadow-lg">
                        <div className="text-[10px] text-[#45a29e] uppercase font-bold mb-1">Mean Absolute Error</div>
                        <div className="text-3xl font-mono text-white">{mae.toFixed(4)}s</div>
                    </div>
                </div>
            )}

            <div className="bg-[#1f2833] rounded-2xl overflow-hidden border border-[#45a29e]/20 shadow-2xl">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-[#0b0c10] text-[#45a29e] text-[10px] uppercase tracking-widest">
                            <th className="p-4 border-b border-[#45a29e]/20">Driver</th>
                            <th className="p-4 border-b border-[#45a29e]/20">Actual Rank</th>
                            <th className="p-4 border-b border-[#45a29e]/20">Predicted Rank</th>
                            <th className="p-4 border-b border-[#45a29e]/20">Actual Time</th>
                            <th className="p-4 border-b border-[#45a29e]/20">Predicted Time</th>
                            <th className="p-4 border-b border-[#45a29e]/20 text-right">Error (Δ)</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[#45a29e]/10">
                        {results?.map((r, i) => (
                            <tr key={r.driver} className="hover:bg-[#66fcf1]/5 transition-colors group">
                                <td className="p-4">
                                    <div className="font-bold text-white group-hover:text-[#66fcf1]">{r.driver.toUpperCase()}</div>
                                    <div className="text-[10px] text-[#45a29e] uppercase font-mono">{r.team}</div>
                                </td>
                                <td className="p-4 font-mono text-xl">{r.actual_rank}</td>
                                <td className={`p-4 font-mono text-xl ${r.predicted_rank === r.actual_rank ? 'text-[#66fcf1]' : 'text-white'}`}>
                                    {r.predicted_rank}
                                </td>
                                <td className="p-4 font-mono text-sm">{r.actual_time}</td>
                                <td className="p-4 font-mono text-sm text-[#45a29e] italic">{r.predicted_time}</td>
                                <td className="p-4 text-right">
                                    <span className={`font-mono text-sm px-2 py-1 rounded bg-[#0b0c10] border ${r.delta_error < 0.1 ? 'border-[#66fcf1] text-[#66fcf1]' : 'border-red-500/30 text-red-400'
                                        }`}>
                                        {r.delta_error.toFixed(3)}s
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {results.length === 0 && !loading && (
                    <div className="p-24 text-center text-[#45a29e] italic uppercase tracking-widest opacity-50">
                        Awaiting historical data analysis...
                    </div>
                )}
            </div>
        </main>
    );
}
