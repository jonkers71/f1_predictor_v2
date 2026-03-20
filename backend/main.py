from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from core.data_engine import F1DataEngine
import logging
import pandas as pd
from datetime import datetime
import numpy as np
import os
import asyncio

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

# Global sync state
_sync_status = {
    "is_syncing": False,
    "progress": 0,
    "stage": "Idle",
    "error": None,
    "last_synced": None,
    "last_sync_results": None
}

# Global retrain state
_retrain_status = {
    "is_training": False,
    "progress": 0,
    "stage": "Idle",
    "error": None,
    "last_trained": None,
    "sessions_used": 0
}

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

# ─────────────────────────────────────────────
# SYNC ENDPOINTS
# ─────────────────────────────────────────────

def _run_sync_background():
    """Background worker that runs the full staged sync pipeline."""
    global _sync_status
    _sync_status["is_syncing"] = True
    _sync_status["error"] = None
    _sync_status["progress"] = 0
    _sync_status["stage"] = "Starting..."

    def on_update(pct: int, stage: str):
        _sync_status["progress"] = pct
        _sync_status["stage"] = stage

    try:
        results = engine.run_full_sync(update_callback=on_update)
        _sync_status["last_synced"] = datetime.now().isoformat()
        _sync_status["last_sync_results"] = results
        _sync_status["progress"] = 100
        _sync_status["stage"] = "Complete"
    except Exception as e:
        _sync_status["error"] = str(e)
        _sync_status["stage"] = f"Failed: {e}"
        logger.error(f"Sync background task failed: {e}")
    finally:
        _sync_status["is_syncing"] = False

@app.post("/sync")
async def sync_data(background_tasks: BackgroundTasks):
    """Triggers a real staged data sync pipeline in the background."""
    global _sync_status
    if _sync_status["is_syncing"]:
        return {"status": "already_running", "message": "Sync is already in progress."}
    background_tasks.add_task(_run_sync_background)
    return {"status": "started", "message": "Sync pipeline started in background."}

@app.get("/sync/status")
async def sync_status():
    """Returns the current sync status including progress and stage."""
    return {
        "is_syncing": _sync_status["is_syncing"],
        "progress": _sync_status["progress"],
        "stage": _sync_status["stage"],
        "error": _sync_status["error"],
        "last_synced": _sync_status["last_synced"],
        "last_sync_results": _sync_status["last_sync_results"]
    }

# ─────────────────────────────────────────────
# DATA HEALTH ENDPOINT
# ─────────────────────────────────────────────

@app.get("/data/health")
async def data_health():
    """Returns a structured health report of the local FastF1 cache."""
    try:
        return engine.check_cache_health()
    except Exception as e:
        logger.error(f"Error checking data health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# MODEL STATUS & RETRAIN ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/model/status")
async def model_status():
    """Returns the status and last training time of the XGBoost models."""
    from core.models import MODELS_DIR
    ranker_path = os.path.join(MODELS_DIR, "ranker.json")
    
    status = "offline"
    last_trained = "never"
    
    if os.path.exists(ranker_path):
        status = "active"
        mtime = os.path.getmtime(ranker_path)
        last_trained = datetime.fromtimestamp(mtime).isoformat()
        
    return {
        "status": status,
        "last_trained": last_trained,
        "feature_version": "2.1 (High-Fidelity Stints)",
        "engine": "XGBoost 2.0.3",
        "training_type": _retrain_status.get("training_type", "synthetic"),
        "sessions_trained_on": _retrain_status.get("sessions_used", 0)
    }

def _run_retrain_background():
    """
    Background worker that retrains the XGBoost models on real cached FastF1 data.
    Falls back to synthetic data if insufficient real sessions are found.
    """
    global _retrain_status
    _retrain_status["is_training"] = True
    _retrain_status["error"] = None
    _retrain_status["progress"] = 0
    _retrain_status["stage"] = "Starting retraining..."
    _retrain_status["sessions_used"] = 0

    try:
        from core.models import F1PredictorModel
        import asyncio

        model = F1PredictorModel()
        model.build_models()

        _retrain_status["progress"] = 5
        _retrain_status["stage"] = "Scanning cache for real sessions..."

        # Collect real training data from cached sessions
        training_rows = []
        sessions_processed = 0

        for yr in [2024, 2025]:
            try:
                schedule = engine.get_event_schedule(yr)
                if schedule is None or schedule.empty:
                    continue
                past_rounds = schedule[schedule['RoundNumber'] > 0]
                total = len(past_rounds)

                for idx, (_, event) in enumerate(past_rounds.iterrows()):
                    round_num = int(event['RoundNumber'])
                    pct = 5 + int(((yr - 2024) * total + idx + 1) / (2 * max(total, 1)) * 70)
                    _retrain_status["progress"] = pct
                    _retrain_status["stage"] = f"Processing {yr} Round {round_num} ({idx+1}/{total})..."

                    for session_type in ['Q', 'R']:
                        try:
                            # Load the target session (actual results)
                            target = engine.get_session(yr, round_num, session_type)
                            actual_results = engine.get_driver_results(target)
                            if actual_results.empty:
                                continue

                            # Load baseline session for features
                            baseline_ids = ['FP3', 'FP2', 'FP1'] if session_type == 'Q' else ['Q', 'FP3', 'FP2']
                            ideal_laps_dict = {}
                            lap_counts_dict = {}
                            consistency_dict = {}
                            long_stint_dict = {}

                            for b_id in baseline_ids:
                                try:
                                    b_session = engine.get_session(yr, round_num, b_id)
                                    i_laps = engine.get_ideal_laps(b_session)
                                    if not i_laps.empty:
                                        ideal_laps_dict = i_laps.set_index('Driver')['IdealLapSeconds'].to_dict()
                                        lap_counts_dict = engine.get_lap_counts(b_session)
                                        consistency_dict = engine.get_lap_consistency(b_session)
                                        if session_type == 'R':
                                            long_stint_dict = engine.get_long_stint_pace(b_session)
                                        break
                                except:
                                    continue

                            if not ideal_laps_dict:
                                continue

                            # Build team ratings and maturities for this session
                            team_ratings = {}
                            maturities = {}
                            sunday_convs = {}
                            for _, row in actual_results.iterrows():
                                team = row['TeamName']
                                if team not in team_ratings:
                                    team_ratings[team] = engine.get_rolling_team_rating(team, yr, round_num)
                                    maturities[team] = engine.get_constructor_maturity(team)
                                    if session_type == 'R':
                                        sunday_convs[team] = engine.get_sunday_conversion_factor(team, years=[yr])

                            session_context = {
                                "results": actual_results.to_dict('records'),
                                "weather": engine.get_weather_summary(target),
                                "session_name": target.name,
                                "ideal_laps": ideal_laps_dict,
                                "team_ratings": team_ratings,
                                "constructor_maturity": maturities,
                                "sunday_conversion": sunday_convs,
                                "lap_counts": lap_counts_dict,
                                "consistency": consistency_dict,
                                "long_stint_pace": long_stint_dict
                            }

                            from core.models import F1PredictorModel as _M
                            _tmp = _M()
                            X = _tmp.prepare_features(session_context)
                            if X.empty:
                                continue

                            # Add actual finishing positions as labels
                            pos_map = {row['Abbreviation']: int(row['Position'])
                                       for _, row in actual_results.iterrows()
                                       if str(row['Position']).isdigit()}
                            y_ranks = []
                            for _, row in actual_results.iterrows():
                                abbr = row['Abbreviation']
                                y_ranks.append(pos_map.get(abbr, 20))

                            X['_rank'] = y_ranks
                            training_rows.append(X)
                            sessions_processed += 1

                        except Exception as e:
                            logger.warning(f"Skipping {yr} R{round_num} {session_type}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error processing {yr}: {e}")

        _retrain_status["progress"] = 78
        _retrain_status["stage"] = f"Building training dataset from {sessions_processed} real sessions..."

        use_real = len(training_rows) >= 5

        if use_real:
            # Combine all real session data
            all_data = pd.concat(training_rows, ignore_index=True)
            y_rank = all_data['_rank'].values
            X_train = all_data.drop(columns=['_rank'])

            # Build group counts (one group per session, infer from rank resets)
            groups = []
            current_group = 1
            for i in range(1, len(y_rank)):
                if y_rank[i] < y_rank[i-1]:
                    groups.append(current_group)
                    current_group = 1
                else:
                    current_group += 1
            groups.append(current_group)

            y_delta = X_train["base_pace_delta"].values + (X_train["stint_slope"].values * 10)
            group_counts = np.array(groups)

            _retrain_status["training_type"] = "real"
            _retrain_status["sessions_used"] = sessions_processed
            logger.info(f"Retraining on {len(X_train)} real rows from {sessions_processed} sessions.")
        else:
            # Fallback to synthetic if not enough real data cached
            logger.warning(f"Only {sessions_processed} real sessions found. Falling back to synthetic training.")
            _retrain_status["stage"] = f"Insufficient real data ({sessions_processed} sessions). Using synthetic fallback..."

            rows = 1000
            data = []
            for _ in range(rows):
                i_delta = np.random.uniform(0, 2.0)
                base_delta = i_delta + np.random.uniform(-0.2, 0.5)
                feat = {
                    "ideal_lap_delta": float(i_delta),
                    "base_pace_delta": float(base_delta),
                    "stint_slope": float(np.random.uniform(0, 0.15)),
                    "stint_variance": float(np.random.uniform(0, 0.5)),
                    "team_strength": float(np.random.uniform(0.1, 1.0)),
                    "reliability": float(np.random.uniform(0.5, 1.0)),
                    "constructor_maturity": float(np.random.choice([0.15, 1.0], p=[0.1, 0.9])),
                    "is_rookie": int(np.random.choice([0, 1], p=[0.8, 0.2])),
                    "sunday_conversion_factor": float(np.random.uniform(-0.02, 0.02)),
                    "track_temp": 30.0,
                    "is_sprint": 0,
                    "has_long_stint": 1
                }
                data.append(feat)
            X_train = pd.DataFrame(data)
            scores = (X_train["base_pace_delta"] * 20 + X_train["stint_slope"] * 100 +
                      X_train["stint_variance"] * 10 + (1.0 - X_train["team_strength"]) * 10 +
                      (1.0 - X_train["reliability"]) * 5 + X_train["is_rookie"] * 2 +
                      (1.0 - X_train["constructor_maturity"]) * 15 - X_train["sunday_conversion_factor"] * 20)
            y_rank = []
            groups = []
            for i in range(0, rows, 20):
                group_scores = scores[i:i+20]
                rank_indices = np.argsort(group_scores)
                ranks = np.zeros(20)
                for r_idx, s_idx in enumerate(rank_indices):
                    ranks[s_idx] = r_idx + 1
                y_rank.extend(ranks)
                groups.append(20)
            y_rank = np.array(y_rank)
            y_delta = X_train["base_pace_delta"].values + (X_train["stint_slope"].values * 10)
            group_counts = np.array(groups)
            _retrain_status["training_type"] = "synthetic"
            _retrain_status["sessions_used"] = 0

        _retrain_status["progress"] = 85
        _retrain_status["stage"] = "Training XGBoost Ranker..."
        model.train_ranker(X_train, y_rank, group_counts)

        _retrain_status["progress"] = 95
        _retrain_status["stage"] = "Training XGBoost Regressor..."
        model.train_regressor(X_train, y_delta)

        _retrain_status["last_trained"] = datetime.now().isoformat()
        _retrain_status["progress"] = 100
        _retrain_status["stage"] = f"Complete. Trained on {'real' if use_real else 'synthetic'} data."
        logger.info(f"Retraining complete. Type: {'real' if use_real else 'synthetic'}, Sessions: {sessions_processed}")

    except Exception as e:
        _retrain_status["error"] = str(e)
        _retrain_status["stage"] = f"Failed: {e}"
        logger.error(f"Retrain background task failed: {e}")
    finally:
        _retrain_status["is_training"] = False

@app.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Triggers a real model retraining using cached FastF1 data."""
    global _retrain_status
    if _retrain_status["is_training"]:
        return {"status": "already_running", "message": "Retraining is already in progress."}
    # Check if there is enough data to retrain on real data
    health = engine.check_cache_health()
    total_cached = health.get("total_sessions_cached", 0)
    background_tasks.add_task(_run_retrain_background)
    return {
        "status": "started",
        "message": f"Retraining started. {total_cached} cached sessions available.",
        "will_use_real_data": total_cached >= 5
    }

@app.get("/model/retrain/progress")
async def retrain_progress():
    """Returns the current model retraining progress."""
    return {
        "is_training": _retrain_status["is_training"],
        "progress": _retrain_status["progress"],
        "stage": _retrain_status["stage"],
        "error": _retrain_status["error"],
        "last_trained": _retrain_status["last_trained"],
        "sessions_used": _retrain_status["sessions_used"],
        "training_type": _retrain_status.get("training_type", "unknown")
    }

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
        
        # Calculate maturities and conversions for backtest
        maturities_dict = {t: 1.0 for t in actual_results['TeamName'].unique()}
        sunday_conv_dict = {t: engine.get_sunday_conversion_factor(t, count=3) for t in actual_results['TeamName'].unique()}

        session_context = {
            "results": actual_results.to_dict('records'),
            "weather": engine.get_weather_summary(session),
            "session_name": session.name,
            "ideal_laps": ideal_laps_dict,
            "team_ratings": team_ratings_dict,
            "constructor_maturity": maturities_dict,
            "sunday_conversion": sunday_conv_dict,
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
            "comparison": comparison
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
        sunday_conversions = {}
        
        data_sources = {
            "type": "none",
            "circuit": circuit_name,
            "sessions_used": [],
            "current_season_adjustment": True,
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
                rating = engine.get_rolling_team_rating(team, year, gp, window=3)
                maturity = engine.get_constructor_maturity(team)
                
                if maturity < 1.0:
                    blend_factor = min(1.0, (gp - 1) / 3.0) 
                    effective_rating = (maturity * (1.0 - blend_factor)) + (rating * blend_factor)
                    team_ratings_dict[team] = effective_rating
                    maturities_dict[team] = maturity
                    data_sources["notes"].append(f"{team}: Maturity blend {int((1-blend_factor)*100)}% Applied")
                else:
                    team_ratings_dict[team] = rating
                    maturities_dict[team] = 1.0

            # Dynamic Rookie Detection (check career race count across 2023-2025)
            abbr = driver["Abbreviation"]
            if abbr not in rookie_flags:
                race_count = engine.get_driver_race_count(abbr)
                rookie_flags[abbr] = race_count < 10
                if rookie_flags[abbr]:
                    data_sources["notes"].append(f"{abbr} flagged as rookie (<10 races)")

            # Sunday Conversion Factor (for Race predictions)
            if session_type == 'R' and team not in sunday_conversions:
                sunday_conversions[team] = engine.get_sunday_conversion_factor(team)
                if abs(sunday_conversions[team]) > 0.005:
                    label = "Sunday car" if sunday_conversions[team] > 0 else "Saturday car"
                    data_sources["notes"].append(f"{team}: {label} ({sunday_conversions[team]:+.1%})")

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
                
                base_time = 90.0
                
                for yr_idx, yr in enumerate(history["years_searched"]):
                    try:
                        h_session = engine.get_session(yr, circuit_name, history_type)
                        h_laps = h_session.laps.pick_quicklaps()
                        if h_laps.empty: continue
                        
                        pole_h = h_laps['LapTime'].min().total_seconds()
                        
                        for drv, hist_delta in history["deltas"].items():
                            found_current = next((d for d in current_grid if d["Abbreviation"] == drv), None)
                            if not found_current:
                                ideal_laps_dict[drv] = base_time + hist_delta
                                continue

                            team_h = h_laps[h_laps['Driver'] == drv]['Team'].iloc[0]
                            team_h_best = h_laps[h_laps['Team'] == team_h]['LapTime'].min().total_seconds()
                            h_deficit = (team_h_best - pole_h) / pole_h
                            
                            if h_deficit <= 0.05: hist_car_rating = 1.0
                            elif h_deficit <= 1.0: hist_car_rating = 1.0 - (h_deficit * 0.3)
                            else: hist_car_rating = max(0.2, 0.7 - ((h_deficit - 1.0) * (0.5 / 1.5)))
                            
                            team_2026 = found_current["TeamName"]
                            rating_2026 = team_ratings_dict.get(team_2026, 0.5)
                            
                            car_slump_penalty = (hist_car_rating - rating_2026) * 4.0
                            adjusted_delta = hist_delta + max(0, car_slump_penalty)
                            ideal_laps_dict[drv] = base_time + adjusted_delta
                            
                        break
                    except: continue

                data_sources["notes"].append("Applied 100% dynamic car-performance decoupling.")
                data_sources["notes"].append("Applied car-performance decoupling to historical times.")

        # 5. Process current grid and Driver Inheritance (Team Switch Penalty)
        final_mapped_laps = {}
        for driver in current_grid:
            abbr = driver["Abbreviation"]
            team = driver["TeamName"]
            rating_curr = team_ratings_dict.get(team, 0.1)
            
            if abbr in ideal_laps_dict:
                final_mapped_laps[abbr] = ideal_laps_dict[abbr]
            else:
                teammate = next((d for d in current_grid if d["TeamName"] == team and d["Abbreviation"] != abbr), None)
                if teammate and teammate["Abbreviation"] in ideal_laps_dict:
                    final_mapped_laps[abbr] = ideal_laps_dict[teammate["Abbreviation"]]
                    data_sources["notes"].append(f"{abbr} inheriting data from {teammate['Abbreviation']}")
                else:
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
            "sunday_conversion": sunday_conversions,
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
                "breakdown": breakdowns[i],
                "confidence": breakdowns[i].get("confidence_score", 100)
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
