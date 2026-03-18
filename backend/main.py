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
    Predicts the upcoming race or qualifying using circuit-specific history
    and the current 2026 driver grid.
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
        data_sources = {
            "type": "none",
            "circuit": circuit_name,
            "sessions_used": [],
            "current_season_adjustment": True,
            "has_long_stint_data": False,
            "notes": []
        }
        
        # === STAGE 1: Try current year practice sessions ===
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
                    for _, r in engine.get_driver_results(b_session).iterrows():
                        team_ratings_dict[r['TeamName']] = engine.get_team_rating(r['TeamName'])
                    source_session = b_session
                    data_sources["type"] = "live_practice"
                    data_sources["sessions_used"].append(f"{year} R{gp} {b_id}")
                    break
            except:
                continue
        
        # Try to get long stint data if available (for race predictions)
        if session_type == 'R':
            for fp_id in ['FP2', 'FP3', 'FP1']:
                try:
                    fp_session = engine.get_session(year, gp, fp_id)
                    stint_data = engine.get_long_stint_pace(fp_session)
                    if stint_data:
                        long_stint_dict = stint_data
                        data_sources["has_long_stint_data"] = True
                        data_sources["sessions_used"].append(f"{year} R{gp} {fp_id} (long stints)")
                        break
                except:
                    continue
        
        # === STAGE 2: If no practice data, use circuit history ===
        if not ideal_laps_dict:
            logger.info(f"No practice data for R{gp}, using circuit history for '{circuit_name}'")
            
            history_type = session_type if session_type in ['Q', 'R'] else 'Q'
            history = engine.get_circuit_history(circuit_name, history_type, [2025, 2024, 2023])
            
            if history["deltas"]:
                ideal_laps_dict = {}
                # Convert deltas back to absolute times (use 90s as a baseline reference)
                base_time = 90.0
                for drv, delta in history["deltas"].items():
                    ideal_laps_dict[drv] = base_time + delta
                
                data_sources["type"] = "circuit_history"
                data_sources["sessions_used"] = history["sessions_found"]
                data_sources["notes"].append(f"Based on {len(history['sessions_found'])} historical sessions at {circuit_name}")
            else:
                data_sources["notes"].append(f"No historical data found for {circuit_name}")
        
        # === STAGE 3: Get the current 2026 grid ===
        current_grid = engine.get_current_grid(year)
        
        if not current_grid:
            # Try previous year grid as last resort
            current_grid = engine.get_current_grid(year - 1)
            if current_grid:
                data_sources["notes"].append(f"Using {year-1} grid (no {year} sessions completed yet)")
        
        if not current_grid:
            return {
                "event": active['name'],
                "round": active['round'],
                "baseline": "No Grid Data",
                "data_sources": data_sources,
                "predictions": []
            }
        
        # === STAGE 4: Map historical data to current grid ===
        # For each driver on the current grid, find their data or use teammate's
        drivers_list = []
        mapped_laps = {}
        
        # Build a team->drivers map from history for teammate inheritance
        team_drivers_history = {}
        for drv, time_val in ideal_laps_dict.items():
            # We need to figure out which team this driver was on (from any session)
            for grid_driver in current_grid:
                if grid_driver["Abbreviation"] == drv:
                    team = grid_driver["TeamName"]
                    if team not in team_drivers_history:
                        team_drivers_history[team] = []
                    team_drivers_history[team].append({"abbr": drv, "time": time_val})
        
        rookies_handled = []
        for driver in current_grid:
            abbr = driver["Abbreviation"]
            team = driver["TeamName"]
            team_ratings_dict[team] = engine.get_team_rating(team)
            
            if abbr in ideal_laps_dict:
                # Driver has historical/practice data
                mapped_laps[abbr] = ideal_laps_dict[abbr]
            else:
                # No data for this driver — try teammate inheritance
                team_history = team_drivers_history.get(team, [])
                if team_history:
                    # Use teammate's average
                    teammate_avg = np.mean([t["time"] for t in team_history])
                    mapped_laps[abbr] = float(teammate_avg)
                    rookies_handled.append(f"{abbr} → teammate avg from {team}")
                else:
                    # No teammate data either — use team rating as proxy
                    # Higher rated team = lower delta
                    t_rating = engine.get_team_rating(team)
                    mapped_laps[abbr] = 90.0 + (1.0 - t_rating) * 2.0  # Scale by rating
                    rookies_handled.append(f"{abbr} → team rating proxy ({team})")
            
            drivers_list.append(driver)
        
        if rookies_handled:
            data_sources["notes"].append(f"Rookies/new drivers: {', '.join(rookies_handled)}")
        
        if not drivers_list:
            return {
                "event": active['name'],
                "round": active['round'],
                "baseline": "No Driver Data",
                "data_sources": data_sources,
                "predictions": []
            }

        # === STAGE 5: Run prediction ===
        from core.models import F1PredictorModel
        model = F1PredictorModel()

        session_context = {
            "results": drivers_list,
            "weather": {"track_temp": 30},
            "session_name": "Qualifying" if session_type == "Q" else "Race",
            "ideal_laps": mapped_laps,
            "team_ratings": team_ratings_dict,
            "lap_counts": lap_counts_dict,
            "consistency": consistency_dict,
            "long_stint_pace": long_stint_dict
        }
        
        X = model.prepare_features(session_context)
        if X.empty:
            return {
                "event": active['name'],
                "round": active['round'],
                "baseline": "Feature Preparation Failed",
                "data_sources": data_sources,
                "predictions": []
            }

        scores, predicted_deltas, breakdowns = model.predict(X, session_type=session_type)
        
        sorted_score_indices = np.argsort(scores)
        predictions = []
        for i in range(len(drivers_list)):
            pred_rank = int(np.where(sorted_score_indices == i)[0][0] + 1)
            predictions.append({
                "rank": pred_rank,
                "driver": drivers_list[i]["FullName"],
                "team": drivers_list[i]["TeamName"],
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
        logger.error(f"Live prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
