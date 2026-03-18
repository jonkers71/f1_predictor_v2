"use client";

import React from 'react';

interface Weights {
  pace: number;
  team: number;
  reliability: number;
  rookie: number;
  long_stint?: number;
}

interface Breakdown {
  pace_score: number;
  team_score: number;
  reliability_score: number;
  rookie_score: number;
  stint_score?: number;
  final_score: number;
  weights: Weights;
}

interface WeightingBreakdownProps {
  breakdown: Breakdown;
  driverName: string;
}

const WeightingBreakdown: React.FC<WeightingBreakdownProps> = ({ breakdown, driverName }) => {
  const { pace_score, team_score, reliability_score, rookie_score, stint_score, weights } = breakdown;
  
  const items = [
    { label: 'Raw Pace', value: pace_score, weight: weights.pace, color: '#66fcf1' },
  ];
  
  // Add long stint only when present (race predictions with practice data)
  if (weights.long_stint && stint_score !== undefined) {
    items.push({ label: 'Long Stint Pace', value: stint_score, weight: weights.long_stint, color: '#4ecdc4' });
  }
  
  items.push(
    { label: 'Team Strength', value: team_score, weight: weights.team, color: '#45a29e' },
    { label: 'Reliability', value: reliability_score, weight: weights.reliability, color: '#f7f1e3' },
    { label: 'Rookie Var', value: rookie_score, weight: weights.rookie, color: '#ff5252' },
  );

  return (
    <div className="bg-[#0b0c10] p-4 rounded-xl border border-[#45a29e]/30 mt-4 animate-in fade-in zoom-in-95 duration-300">
      <div className="flex justify-between items-center mb-3">
        <h4 className="text-[10px] font-bold text-[#45a29e] uppercase tracking-widest">
          Weighting Analysis: {driverName}
        </h4>
        <span className="text-[10px] text-white/50 font-mono italic">Lower is better</span>
      </div>
      
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.label} className="space-y-1">
            <div className="flex justify-between text-[10px] uppercase font-mono">
              <span style={{ color: item.color }}>{item.label} ({item.weight}%)</span>
              <span className="text-white font-bold">{item.value.toFixed(2)}</span>
            </div>
            <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{ 
                  width: `${Math.min((item.value / 70) * 100, 100)}%`, 
                  backgroundColor: item.color,
                  boxShadow: `0 0 10px ${item.color}44`
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-4 pt-3 border-t border-[#45a29e]/10 flex justify-between items-center">
        <span className="text-[10px] text-[#45a29e] uppercase font-bold">Aggregated Score</span>
        <span className="text-lg font-black text-[#66fcf1] font-mono">{breakdown.final_score.toFixed(2)}</span>
      </div>
    </div>
  );
};

export default WeightingBreakdown;
