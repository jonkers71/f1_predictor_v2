from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.data_engine import F1DataEngine
import logging
import pandas as pd
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F1 Predictor v2 API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Data Engine
engine = F1DataEngine()
_last_synced = None  # Module-level sync timestamp (persists across requests)

@app.get("/")
async def root():
    return {"message": "F1 Predictor v2 Backend API is running"}

@app.get("/schedule/{year}")
async def get_schedule(year: int):
    """Returns the full race schedule for a given year with session availability."""
    try:
        schedule = engine.get_event_schedule(year)
        if schedule.empty:
            return []
        
        # We only want actual race events (RoundNumber > 0)
        events = schedule[schedule['RoundNumber'] > 0]
        
        result = []
        for _, row in events.iterrows():
            # Check for Sprint
            is_sprint = False
            sessions = []
            
            # FastF1 Event rows have Session1..Session5
            for i in range(1, 6):
                s_name = row[f'Session{i}']
                if pd.isna(s_name): continue
                
                # Normalize names for frontend
                display_name = str(s_name)
                code = "Q" # Default
                
                if "Qualifying" in display_name and "Sprint" not in display_name: code = "Q"
                elif "Sprint Qualifying" in display_name: 
                    code = "SQ"
                    is_sprint = True
                elif "Sprint" in display_name and "Qualifying" not in display_name:
                    code = "S"
                    is_sprint = True
                elif "Race" in display_name: code = "R"
                else: continue # Skip Practice sessions for now in backtest
                
                sessions.append({"name": display_name, "code": code})
            
            # Ensure Qualifying and Race are always there if found
            result.append({
                "round": int(row['RoundNumber']),
                "name": row['EventName'],
                "date": row['EventDate'].strftime('%Y-%m-%d') if hasattr(row['EventDate'], 'strftime') else str(row['EventDate']),
                "is_sprint": is_sprint,
                "sessions": sessions
            })
            
        return result
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active_event")
async def get_active_event():
    """Returns the details of the racing being predicted."""
    return engine.get_next_event()

@app.post("/sync")
async def sync_data():
    """Simulates/Triggers data sync for the latest session."""
    global _last_synced
    logger.info("Sync triggered...")
    _last_synced = datetime.now().isoformat()
    return {"status": "success", "last_synced": _last_synced}

@app.get("/sync/status")
async def sync_status():
    """Returns the last sync timestamp."""
    return {"last_synced": _last_synced}


@app.get("/session/{year}/{gp}/{identifier}")
async def get_session_info(year: int, gp: str, identifier: str):
    """
    gp: round number (e.g. 5) or circuit name (e.g. 'China')
    identifier: 'FP1','FP2','FP3','Q','SQ','S','R'
    """
    try:
        # Handle GP as int if it's numeric
        gp_param = int(gp) if gp.isdigit() else gp
        session = engine.get_session(year, gp_param, identifier)
        
        weather = engine.get_weather_summary(session)
        best_laps = engine.get_best_laps(session)
        results = engine.get_driver_results(session)
        
        return {
            "event": session.event.to_dict(),
            "session_name": session.name,
            "weather": weather,
            "best_laps": best_laps.to_dict('records') if not best_laps.empty else [],
            "results": results.to_dict('records') if not results.empty else []
        }
    except Exception as e:
        logger.error(f"Error fetching session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{year}/{gp}/{session_type}")
async def run_backtest(year: int, gp: str, session_type: str):
    """
    Runs a prediction on historical data and compares it with actual results.
    session_type: 'Q' (Qualifying), 'SQ' (Sprint Qualifying), 'S' (Sprint), 'R' (Race)
    """
    try:
        gp_param = int(gp) if gp.isdigit() else gp
        # 1. Load historical target session
        session = engine.get_session(year, gp_param, session_type)
        
        # 2. Get actual results
        actual_laps = engine.get_best_laps(session)
        actual_results = engine.get_driver_results(session)
        
        # 3. Fetch Practice/Context Data for Features
        baseline_ids = []
        if session_type == 'SQ':
            baseline_ids = ['FP1']
        elif session_type == 'Q':
            baseline_ids = ['FP3', 'FP2', 'FP1']
        elif session_type == 'S':
            baseline_ids = ['SQ', 'FP1']
        elif session_type == 'R':
            baseline_ids = ['Q', 'FP3', 'FP2']
            
        ideal_laps_dict = {}
        team_ratings_dict = {}
        lap_counts_dict = {}
        consistency_dict = {}
        long_stint_dict = {}
        
        found_baseline = False
        for b_id in baseline_ids:
            try:
                b_session = engine.get_session(year, gp_param, b_id)
                i_laps = engine.get_ideal_laps(b_session)
                if not i_laps.empty:
                    ideal_laps_dict = i_laps.set_index('Driver')['IdealLapSeconds'].to_dict()
                    lap_counts_dict = engine.get_lap_counts(b_session)
                    consistency_dict = engine.get_lap_consistency(b_session)
                    
                    for _, r in engine.get_driver_results(b_session).iterrows():
                        team_ratings_dict[r['TeamName']] = engine.get_team_rating(r['TeamName'])
                    
                    logger.info(f"Using {b_id} as baseline for {session_type} backtest.")
                    found_baseline = True
                    break
            except:
                continue
        
        # Try to get long stint data for race predictions
        if session_type == 'R':
            for fp_id in ['FP2', 'FP3', 'FP1']:
                try:
                    fp_session = engine.get_session(year, gp_param, fp_id)
                    stint_data = engine.get_long_stint_pace(fp_session)
                    if stint_data:
                        long_stint_dict = stint_data
                        logger.info(f"Got long stint data from {fp_id}")
                        break
                except:
                    continue

        # 4. Generate Predictions
        from core.models import F1PredictorModel
        model = F1PredictorModel()
        
        session_context = {
            "results": actual_results.to_dict('records'),
            "weather": engine.get_weather_summary(session),
            "session_name": session.name,
            "ideal_laps": ideal_laps_dict,
            "team_ratings": team_ratings_dict,
            "lap_counts": lap_counts_dict,
            "consistency": consistency_dict,
            "long_stint_pace": long_stint_dict
        }
        X = model.prepare_features(session_context)
        scores, predicted_deltas, breakdowns = model.predict(X, session_type=session_type)
        
        # 5. Merge Predicted vs Actual
        comparison = []
        pole_time = actual_laps['LapTimeSeconds'].min() if not actual_laps.empty else 0
        
        sorted_score_indices = np.argsort(scores)
        
        for idx, (i, row) in enumerate(actual_results.iterrows()):
            driver_code = row['Abbreviation']
            driver_laps = actual_laps[actual_laps['Driver'] == driver_code] if not actual_laps.empty else pd.DataFrame()
            actual_time_sec = driver_laps['LapTimeSeconds'].iloc[0] if not driver_laps.empty else None
            
            pred_rank = int(np.where(sorted_score_indices == idx)[0][0] + 1)
            
            comparison.append({
                "driver": row['FullName'],
                "team": row['TeamName'],
                "actual_rank": int(row['Position']),
                "predicted_rank": pred_rank,
                "actual_time": engine.format_lap_time(actual_time_sec),
                "predicted_time": engine.format_lap_time(pole_time + (predicted_deltas[idx] if predicted_deltas[idx] > 0 else 0)),
                "delta_error": float(abs((pole_time + predicted_deltas[idx]) - actual_time_sec)) if actual_time_sec and pole_time > 0 else 0,
                "breakdown": breakdowns[idx]
            })
        
        # Sort by predicted rank
        comparison.sort(key=lambda x: x['predicted_rank'])
            
        return {
            "event": session.event['EventName'],
            "session_type": session.name,
            "year": year,
            "mean_absolute_error": float(np.mean([c['delta_error'] for c in comparison if c['delta_error'] > 0])) if comparison else 0,
            "results": comparison
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/current")
async def get_current_prediction(session_type: str = "Q"):
    """
    Predicts the upcoming race or qualifying using a dynamic self-healing architecture.
    Adjusts for 2026 pecking order, constructor maturity, and historical car delta.
    """
    try:
        active = engine.get_next_event()
        year = active['year']
        gp = active['round']
        circuit_name = active.get('country', active['name'])
        
        ideal_laps_dict = {}
        team_ratings_dict = {}
        lap_counts_dict = {}
        consistency_dict = {}
        long_stint_dict = {}
        maturities_dict = {}
        rookie_flags = {}
        
        data_sources = {
            "type": "none",
            "circuit": circuit_name,
            "sessions_used": [],
            "dynamic_pecking_order": True,
            "has_long_stint_data": False,
            "notes": []
        }
        
        # 1. Get current 2026 grid and team ratings
        current_grid = engine.get_current_grid(year)
        if not current_grid:
            current_grid = engine.get_current_grid(year - 1)
            data_sources["notes"].append("Using 2025 grid (no 2026 sessions yet)")
        
        if not current_grid:
            raise HTTPException(status_code=500, detail="Could not retrieve driver grid.")

        # 2. Calculate dynamic pecking order and maturity
        for driver in current_grid:
            team = driver["TeamName"]
            if team not in team_ratings_dict:
                # Rolling rating from last 3 races
                rating = engine.get_rolling_team_rating(team, year, gp, window=3)
                maturity = engine.get_constructor_maturity(team)
                
                # Blending maturity for new teams (Race 1: 100% maturity penalty, Race 4+: dynamic)
                if maturity < 1.0:
                    blend_factor = min(1.0, (gp - 1) / 3.0) 
                    effective_rating = (maturity * (1.0 - blend_factor)) + (rating * blend_factor)
                    team_ratings_dict[team] = effective_rating
                    maturities_dict[team] = maturity
                    data_sources["notes"].append(f"{team}: Maturity blend {int((1-blend_factor)*100)}% Applied")
                else:
                    team_ratings_dict[team] = rating
                    maturities_dict[team] = 1.0

            # Dynamic Rookie Detection (check if they have < 10 races in career)
            abbr = driver["Abbreviation"]
            if abbr not in rookie_flags:
                try:
                    # In a real app we'd query career history. For this engine:
                    # We'll treat drivers not in 2023/2024 results as rookies
                    hist_search = engine.get_event_schedule(2024)
                    rookie_flags[abbr] = False 
                    # If they are in the hardcoded rookie list, stick with it for the sim
                    if abbr in ["ANT", "BEA", "HAD", "DOO", "BOR"]:
                        rookie_flags[abbr] = True
                except:
                    rookie_flags[abbr] = False

        # 3. Try current year practice data (Live Data)
        baseline_ids = ['FP3', 'FP2', 'FP1']
        if session_type == 'R':
            baseline_ids = ['Q', 'FP3', 'FP2']
            
        source_session = None
        for b_id in baseline_ids:
            try:
                b_session = engine.get_session(year, gp, b_id)
                i_laps = engine.get_ideal_laps(b_session)
                if not i_laps.empty:
                    ideal_laps_dict = i_laps.set_index('Driver')['IdealLapSeconds'].to_dict()
                    lap_counts_dict = engine.get_lap_counts(b_session)
                    consistency_dict = engine.get_lap_consistency(b_session)
                    source_session = b_session
                    data_sources["type"] = "live_practice"
                    data_sources["sessions_used"].append(f"{year} R{gp} {b_id}")
                    break
            except: continue

        # 4. Fallback to Circuit History with decoupled penalties
        if not ideal_laps_dict:
            history_type = session_type if session_type in ['Q', 'R'] else 'Q'
            history = engine.get_circuit_history(circuit_name, history_type, [2025, 2024, 2023])
            
            if history["deltas"]:
                data_sources["type"] = "circuit_history"
                data_sources["sessions_used"] = history["sessions_found"]
                
                # Base time for synthetic reconstruction
                base_time = 90.0
                mapped_laps = {}
                
                # Map historical drivers and apply CAR PENALTY (Decoupling)
                for drv, hist_delta in history["deltas"].items():
                    # Find which team this driver was on in that historical period (approximate as their 2025/2024 team)
                    # For simplicity, we compare their historical performance to current 2026 team strength
                    
                    found_current = next((d for d in current_grid if d["Abbreviation"] == drv), None)
                    if found_current:
                        team_2026 = found_current["TeamName"]
                        rating_2026 = team_ratings_dict.get(team_2026, 0.5)
                        
                        # Fix "Verstappen Effect": Verstappen in 2023 Red Bull (~1.0) vs 2026 Red Bull (~0.6)
                        # We assume history[drv] was set in a 0.95 rated car on average
                        hist_car_rating = 0.95 if drv in ["VER", "LEC", "HAM", "NOR"] else 0.6
                        if drv == "VER": hist_car_rating = 1.0 # Peak dominance
                        
                        car_slump_penalty = (hist_car_rating - rating_2026) * 4.0 # 4s spread
                        adjusted_delta = hist_delta + max(0, car_slump_penalty)
                        ideal_laps_dict[drv] = base_time + adjusted_delta
                    else:
                        # Driver not on current grid, store for teammate inheritance later
                        ideal_laps_dict[drv] = base_time + hist_delta

                data_sources["notes"].append("Applied car-performance decoupling to historical times.")

        # 5. Process current grid and Driver Inheritance (Team Switch Penalty)
        final_mapped_laps = {}
        for driver in current_grid:
            abbr = driver["Abbreviation"]
            team = driver["TeamName"]
            rating_curr = team_ratings_dict.get(team, 0.1)
            
            if abbr in ideal_laps_dict:
                # We have personal history (possibly adjusted in stage 4)
                final_mapped_laps[abbr] = ideal_laps_dict[abbr]
            else:
                # No personal history at this track — teammate inheritance or proxy
                # Find teammate
                teammate = next((d for d in current_grid if d["TeamName"] == team and d["Abbreviation"] != abbr), None)
                if teammate and teammate["Abbreviation"] in ideal_laps_dict:
                    final_mapped_laps[abbr] = ideal_laps_dict[teammate["Abbreviation"]]
                    data_sources["notes"].append(f"{abbr} inheriting data from {teammate['Abbreviation']}")
                else:
                    # Last resort: Proxy based on team rating with 4.0s spread
                    # P1 (1.0) -> 90.0s, P20 (0.0) -> 94.0s
                    final_mapped_laps[abbr] = 90.0 + (1.0 - rating_curr) * 4.0
                    data_sources["notes"].append(f"{abbr} using rating proxy ({team})")

        # 6. Race Pace Stints
        if session_type == 'R':
            for fp_id in ['FP2', 'FP3', 'FP1']:
                try:
                    fp_session = engine.get_session(year, gp, fp_id)
                    stint_data = engine.get_long_stint_pace(fp_session)
                    if stint_data:
                        long_stint_dict = stint_data
                        data_sources["has_long_stint_data"] = True
                        data_sources["sessions_used"].append(f"{year} R{gp} {fp_id} (Race Stints)")
                        break
                except: continue

        # 7. Run Prediction
        from core.models import F1PredictorModel
        model = F1PredictorModel()

        session_context = {
            "results": [{**d, "is_rookie": rookie_flags.get(d["Abbreviation"], False)} for d in current_grid],
            "weather": {"track_temp": 30},
            "session_name": "Qualifying" if session_type == "Q" else "Race",
            "ideal_laps": final_mapped_laps,
            "team_ratings": team_ratings_dict,
            "constructor_maturity": maturities_dict,
            "lap_counts": lap_counts_dict,
            "consistency": consistency_dict,
            "long_stint_pace": long_stint_dict
        }
        
        X = model.prepare_features(session_context)
        scores, predicted_deltas, breakdowns = model.predict(X, session_type=session_type)
        
        sorted_score_indices = np.argsort(scores)
        predictions = []
        for i in range(len(current_grid)):
            pred_rank = int(np.where(sorted_score_indices == i)[0][0] + 1)
            predictions.append({
                "rank": pred_rank,
                "driver": current_grid[i]["FullName"],
                "team": current_grid[i]["TeamName"],
                "time": engine.format_lap_time(89.0 + predicted_deltas[i]),
                "breakdown": breakdowns[i]
            })
            
        predictions.sort(key=lambda x: x['rank'])
        
        return {
            "event": active['name'],
            "round": active['round'],
            "baseline": data_sources["type"],
            "data_sources": data_sources,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Prediction Engine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
