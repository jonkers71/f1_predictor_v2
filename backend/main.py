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
    import time
    # This would normally pull from FastF1
    # For now, we simulate a multi-step progress for the UI
    logger.info("Sync triggered...")
    return {"status": "success", "last_synced": datetime.now().isoformat()}

@app.get("/schedule/{year}")
async def get_schedule(year: int):
    try:
        schedule = engine.get_event_schedule(year)
        # Convert DataFrame to list of dicts
        # FastF1 schedule has special types, convert to strings for JSON
        return schedule.astype(str).to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        # Determine which session to use as the 'Performance Baseline'
        baseline_ids = []
        if session_type == 'SQ':
            baseline_ids = ['FP1']
        elif session_type == 'Q':
            baseline_ids = ['FP3', 'FP2', 'FP1'] # Latest available
        elif session_type == 'S':
            baseline_ids = ['SQ', 'FP1']
        elif session_type == 'R':
            baseline_ids = ['Q', 'FP3', 'FP2']
            
        ideal_laps_dict = {}
        team_ratings_dict = {}
        lap_counts_dict = {}
        consistency_dict = {}
        
        found_baseline = False
        for b_id in baseline_ids:
            try:
                b_session = engine.get_session(year, gp_param, b_id)
                i_laps = engine.get_ideal_laps(b_session)
                if not i_laps.empty:
                    ideal_laps_dict = i_laps.set_index('Driver')['IdealLapSeconds'].to_dict()
                    lap_counts_dict = engine.get_lap_counts(b_session)
                    consistency_dict = engine.get_lap_consistency(b_session)
                    
                    # Also get team ratings from this session's lineup
                    for _, r in engine.get_driver_results(b_session).iterrows():
                        team_ratings_dict[r['TeamName']] = engine.get_team_rating(r['TeamName'])
                    
                    logger.info(f"Using {b_id} as baseline for {session_type} backtest.")
                    found_baseline = True
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
            "consistency": consistency_dict
        }
        X = model.prepare_features(session_context)
        scores, predicted_deltas = model.predict(X)
        
        # 5. Merge Predicted vs Actual
        comparison = []
        # Fallback if no laps (e.g. DNS)
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
                "delta_error": float(abs((pole_time + predicted_deltas[idx]) - actual_time_sec)) if actual_time_sec and pole_time > 0 else 0
            })
            
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
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
