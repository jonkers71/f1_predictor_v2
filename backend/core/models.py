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

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates predictions for a session.
        Uses a Weighted Scoring Engine as a high-fidelity predictor.
        Returns: (predicted_ranks, predicted_deltas)
        """
        # Feature names check
        if "ideal_lap_delta" not in X.columns:
            # Fallback for old feature sets
            scores = (X["avg_practice_delta"] * 70) + (0.5 * 30)
        else:
            # PERFORMANCE FORMULA:
            # Raw Pace (Ideal Delta) is 75% of the rank
            # Team Engine/Qualy Mode Strength is 25% of the rank
            # We want LOW score for better rank.
            scores = (X["ideal_lap_delta"] * 75) + ((1.0 - X["team_strength"]) * 25)

        # Ranks are the positions of the scores
        # We return the RAW scores to be sorted in the caller or sort them here
        ranks = scores.values
        
        # Predict delta - assume Qualifying is ~0.8s faster than Practice Ideal
        # but deltas between drivers remain relative
        predicted_deltas = (X["ideal_lap_delta"] if "ideal_lap_delta" in X.columns else X["avg_practice_delta"]).values
        
        return ranks, predicted_deltas

    def prepare_features(self, session_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts session data into a feature matrix for prediction.
        Includes Reliability Index (consistency + volume).
        """
        drivers = session_data.get("results", [])
        ideal_laps = session_data.get("ideal_laps", {})
        team_ratings = session_data.get("team_ratings", {})
        lap_counts = session_data.get("lap_counts", {})
        consistency = session_data.get("consistency", {})
        
        # Rookies 2025/2026 for variance penalty
        rookies = ["ANT", "BEA", "HAD", "DOO", "BOR"]
        
        # Calculate field min for deltas
        times = [t for t in ideal_laps.values() if t is not None]
        min_ideal_time = min(times) if times else 0
        
        features_list = []
        for driver in drivers:
            abbr = driver.get("Abbreviation")
            team = driver.get("TeamName")
            i_time = ideal_laps.get(abbr)
            
            i_delta = (i_time - min_ideal_time) if i_time and min_ideal_time else 0.8
            t_rating = team_ratings.get(team, 0.5)
            
            # Reliability metrics (0-1, higher is better)
            laps = lap_counts.get(abbr, 0)
            # Logarithmic scale for lap count (max 20 laps)
            volume_score = min(laps / 15.0, 1.0) 
            
            # Consistency penalty (low stdev is good)
            raw_cons = consistency.get(abbr, 1.0)
            cons_score = max(0, 1.0 - (raw_cons / 1.5)) 
            
            feat = {
                "ideal_lap_delta": float(i_delta),
                "team_strength": float(t_rating),
                "reliability": float((volume_score * 0.4) + (cons_score * 0.6)),
                "is_rookie": 1 if abbr in rookies else 0,
                "track_temp": float(session_data.get("weather", {}).get("track_temp", 30)),
                "is_sprint": 1 if "Sprint" in session_data.get("session_name", "") else 0
            }
            features_list.append(feat)
        
        return pd.DataFrame(features_list)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts Qualifying ranks and lap time deltas.
        """
        # ADVANCED PERFORMANCE FORMULA:
        # 1. Base Pace (Ideal Delta) - 60%
        # 2. Team Latent Pace - 20%
        # 3. Reliability Index (Volume/Consistency) - 15%
        # 4. Rookie Variance - 5% penalty
        
        pace_score = X["ideal_lap_delta"] * 65
        latent_score = (1.0 - X["team_strength"]) * 20
        reliability_penalty = (1.0 - X["reliability"]) * 10
        rookie_penalty = X["is_rookie"] * 5
        
        final_scores = pace_score + latent_score + reliability_penalty + rookie_penalty
        
        # Predicted Rankings
        ranks = final_scores.values
        
        # Predicted Deltas (with small reliability correction)
        # Unreliable drivers (low reliability) tend to drop time in Qualy
        base_delta = X["ideal_lap_delta"].values
        correction = (1.0 - X["reliability"].values) * 0.2
        predicted_deltas = base_delta + correction
        
        return ranks, predicted_deltas

if __name__ == "__main__":
    predictor = F1PredictorModel()
    predictor.build_models()
    print("Models ready.")
