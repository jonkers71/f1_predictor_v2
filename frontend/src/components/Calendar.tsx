"use client";

import { useState, useEffect } from "react";

interface Event {
    name: string;
    round: number;
    date: string;
    type: "Sprint" | "Standard";
}

export default function Calendar() {
    const [events, setEvents] = useState<Event[]>([]);

    useEffect(() => {
        // Fetch real schedule for 2026
        fetch("http://localhost:8000/schedule/2026")
            .then((res) => res.json())
            .then((data) => {
                const formatted = data.slice(0, 5).map((e: any) => ({
                    name: e.name.toUpperCase(),
                    round: e.round,
                    date: e.date,
                    type: e.is_sprint ? 'Sprint' : 'Standard'
                }));
                setEvents(formatted);
            })
            .catch((err) => console.error("Error fetching schedule:", err));
    }, []);

    return (
        <div className="bg-[#1f2833] rounded-2xl p-6 border border-[#45a29e]/20">
            <h3 className="text-sm font-bold text-[#45a29e] uppercase tracking-widest mb-6 flex items-center gap-2">
                Upcoming Races
                <span className="flex-1 h-[1px] bg-[#45a29e]/20"></span>
            </h3>

            <div className="space-y-6">
                {events.map((event, index) => (
                    <div key={`${event.round}-${event.name}-${index}`} className="relative pl-4 border-l-2 border-[#45a29e]/20 hover:border-[#66fcf1] transition-colors group">
                        <div className="flex justify-between items-start">
                            <div>
                                <div className="text-xs text-[#45a29e] font-mono">
                                    {event.round === 0 ? 'PRE-SEASON TESTING' : `ROUND ${event.round}`}
                                </div>
                                <div className="text-lg font-bold text-white group-hover:text-[#66fcf1] transition-colors">
                                    {event.name}
                                </div>
                            </div>
                            <div className={`text-[10px] px-2 py-0.5 rounded border ${event.type === 'Sprint'
                                ? 'border-[#ff3e3e] text-[#ff3e3e] bg-[#ff3e3e]/10'
                                : 'border-[#45a29e] text-[#45a29e]'
                                }`}>
                                {event.type.toUpperCase()}
                            </div>
                        </div>
                        <div className="text-xs text-[#c5c6c7] mt-1 italic">
                            {event.date}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
