import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
os.makedirs(MODELS_DIR, exist_ok=True)

class F1PredictorModel:
    def __init__(self):
        self.ranker = None
        self.regressor = None
        self.ranker_path = os.path.join(MODELS_DIR, "ranker.json")
        self.regressor_path = os.path.join(MODELS_DIR, "regressor.json")
        self.feature_names = [
            "avg_practice_delta", "practice_consistency", "track_temp", 
            "air_temp", "is_sprint", "tyre_life", "team_momentum"
        ]

    def build_models(self):
        """Initializes the XGBoost models."""
        # Learning to Rank (LTR) model
        self.ranker = xgb.XGBRanker(
            objective="rank:ndcg",
            eta=0.1,
            gamma=1.0,
            min_child_weight=0.1,
            max_depth=6,
            n_estimators=100
        )
        
        # Lap time delta regressor
        self.regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=5,
            eta=0.1
        )
        logger.info("XGBoost models initialized (empty).")

    def train_ranker(self, X: pd.DataFrame, y: pd.Series, groups: np.array):
        """
        Trains the ranker.
        X: Features
        y: Rank positions
        groups: Number of drivers per session (for XGBoost group tracking)
        """
        logger.info(f"Training ranker on {len(X)} rows with {len(groups)} groups.")
        self.ranker.fit(X, y, group=groups)
        self.ranker.save_model(self.ranker_path)

    def train_regressor(self, X: pd.DataFrame, y: pd.Series):
        """Trains the lap time delta regressor."""
        logger.info(f"Training regressor on {len(X)} rows.")
        self.regressor.fit(X, y)
        self.regressor.save_model(self.regressor_path)


    def prepare_features(self, session_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts session data into a feature matrix for prediction.
        Includes Reliability Index (consistency + volume) and long stint pace.
        """
        drivers = session_data.get("results", [])
        ideal_laps = session_data.get("ideal_laps", {})
        team_ratings = session_data.get("team_ratings", {})
        lap_counts = session_data.get("lap_counts", {})
        consistency = session_data.get("consistency", {})
        long_stint = session_data.get("long_stint_pace", {})
        
        # Rookies 2025/2026 for variance penalty
        rookies = ["ANT", "BEA", "HAD", "DOO", "BOR"]
        
        # Calculate field min for deltas
        times = [t for t in ideal_laps.values() if t is not None]
        min_ideal_time = min(times) if times else 0
        
        # Calculate field min for long stint
        stint_times = [t for t in long_stint.values() if t is not None]
        min_stint_time = min(stint_times) if stint_times else 0
        
        features_list = []
        for driver in drivers:
            abbr = driver.get("Abbreviation")
            team = driver.get("TeamName")
            i_time = ideal_laps.get(abbr)
            stint_time = long_stint.get(abbr)
            
            i_delta = (i_time - min_ideal_time) if i_time and min_ideal_time else 0.8
            stint_delta = (stint_time - min_stint_time) if stint_time and min_stint_time else i_delta * 1.2  # Estimate from quali pace
            t_rating = team_ratings.get(team, 0.5)
            
            # Reliability metrics (0-1, higher is better)
            laps = lap_counts.get(abbr, 0)
            volume_score = min(laps / 15.0, 1.0)
            
            # Consistency penalty (low stdev is good)
            raw_cons = consistency.get(abbr, 1.0)
            cons_score = max(0, 1.0 - (raw_cons / 1.5))
            
            feat = {
                "ideal_lap_delta": float(i_delta),
                "long_stint_delta": float(stint_delta),
                "team_strength": float(t_rating),
                "reliability": float((volume_score * 0.4) + (cons_score * 0.6)),
                "is_rookie": 1 if abbr in rookies else 0,
                "track_temp": float(session_data.get("weather", {}).get("track_temp", 30)),
                "is_sprint": 1 if "Sprint" in session_data.get("session_name", "") else 0,
                "has_long_stint": 1 if stint_time is not None else 0
            }
            features_list.append(feat)
        
        return pd.DataFrame(features_list)

    def predict(self, X: pd.DataFrame, session_type: str = "Q") -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        Predicts Qualifying or Race ranks and lap time deltas.
        Returns: (ranks, predicted_deltas, component_breakdown)
        """
        # Session-specific weighting
        if session_type == "R":
            # RACE PREDICTION FORMULA:
            # Long stint pace is king for race predictions
            has_stint_data = X["has_long_stint"].sum() > 0
            if has_stint_data:
                # With long stint data available
                w_stint, w_pace, w_team, w_reliability, w_rookie = 35, 20, 25, 15, 5
            else:
                # Fallback: no stint data, use single lap pace more heavily
                w_stint, w_pace, w_team, w_reliability, w_rookie = 0, 45, 30, 20, 5
        else:
            # QUALIFYING PACE FORMULA:
            w_stint, w_pace, w_team, w_reliability, w_rookie = 0, 65, 20, 10, 5
        
        pace_scores = X["ideal_lap_delta"] * w_pace
        stint_scores = X["long_stint_delta"] * w_stint if w_stint > 0 else pace_scores * 0
        latent_scores = (1.0 - X["team_strength"]) * w_team
        reliability_penalties = (1.0 - X["reliability"]) * w_reliability
        rookie_penalties = X["is_rookie"] * w_rookie
        
        final_scores = pace_scores + stint_scores + latent_scores + reliability_penalties + rookie_penalties
        
        # Build breakdown for transparency
        breakdown = []
        for i in range(len(X)):
            bd = {
                "pace_score": float(pace_scores.iloc[i]),
                "team_score": float(latent_scores.iloc[i]),
                "reliability_score": float(reliability_penalties.iloc[i]),
                "rookie_score": float(rookie_penalties.iloc[i]),
                "final_score": float(final_scores.iloc[i]),
                "weights": {
                    "pace": w_pace,
                    "team": w_team,
                    "reliability": w_reliability,
                    "rookie": w_rookie
                }
            }
            if w_stint > 0:
                bd["stint_score"] = float(stint_scores.iloc[i])
                bd["weights"]["long_stint"] = w_stint
            breakdown.append(bd)
        
        # Predicted Rankings (lower score is better)
        ranks = final_scores.values
        
        # Predicted Deltas
        if session_type == "R" and w_stint > 0:
            base_delta = X["long_stint_delta"].values
        else:
            base_delta = X["ideal_lap_delta"].values
        correction = (1.0 - X["reliability"].values) * 0.2
        predicted_deltas = base_delta + correction
        
        return ranks, predicted_deltas, breakdown

if __name__ == "__main__":
    predictor = F1PredictorModel()
    predictor.build_models()
    print("Models ready.")
