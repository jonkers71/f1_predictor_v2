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
            "ideal_lap_delta", "base_pace_delta", "stint_slope", "stint_variance",
            "team_strength", "reliability", "constructor_maturity", "is_rookie",
            "sunday_conversion_factor", "track_temp", "is_sprint", "has_long_stint"
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
        long_stint = session_data.get("long_stint_pace", {}) # Now {driver: {slope, intercept, variance}}
        maturities = session_data.get("constructor_maturity", {})
        sunday_conv = session_data.get("sunday_conversion", {})
        
        # Calculate field min for deltas
        times = [t for t in ideal_laps.values() if t is not None]
        min_ideal_time = min(times) if times else 0
        
        # Calculate field min for base pace (intercept)
        intercepts = [v["intercept"] for v in long_stint.values() if v and "intercept" in v]
        min_intercept = min(intercepts) if intercepts else 0
        
        features_list = []
        for driver in drivers:
            abbr = driver.get("Abbreviation")
            team = driver.get("TeamName")
            i_time = ideal_laps.get(abbr)
            stint_profile = long_stint.get(abbr) # {slope, intercept, variance}
            
            i_delta = (i_time - min_ideal_time) if i_time and min_ideal_time else 1.0
            
            # Extract high-fidelity stint metrics
            if stint_profile:
                base_pace_delta = (stint_profile["intercept"] - min_intercept) if min_intercept else i_delta
                stint_slope = stint_profile["slope"]
                stint_variance = stint_profile["variance"]
                has_stint = 1
            else:
                # Fallback: assume average wear (0.05s/lap) and high variance if no data
                base_pace_delta = i_delta * 1.1
                stint_slope = 0.05
                stint_variance = 0.5
                has_stint = 0
                
            t_rating = team_ratings.get(team, 0.5)
            c_maturity = maturities.get(team, 1.0)
            conversion = sunday_conv.get(team, 0.0)
            
            # Reliability metrics (0-1, higher is better)
            laps = lap_counts.get(abbr, 0)
            volume_score = min(laps / 15.0, 1.0)
            cons_score = max(0, 1.0 - (consistency.get(abbr, 1.0) / 1.5))
            
            reliability_score = ((volume_score * 0.4) + (cons_score * 0.6)) * c_maturity
            
            feat = {
                "ideal_lap_delta": float(i_delta),
                "base_pace_delta": float(base_pace_delta),
                "stint_slope": float(stint_slope),
                "stint_variance": float(stint_variance),
                "team_strength": float(t_rating),
                "reliability": float(reliability_score),
                "constructor_maturity": float(c_maturity),
                "is_rookie": 1 if driver.get("is_rookie") else 0,
                "sunday_conversion_factor": float(conversion),
                "track_temp": float(session_data.get("weather", {}).get("track_temp", 30)),
                "is_sprint": 1 if "Sprint" in session_data.get("session_name", "") else 0,
                "has_long_stint": has_stint
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
            # RACE PREDICTION FORMULA
            # Increased weight on slope, variance, and sunday conversion
            w_base, w_slope, w_variance, w_conv = 20, 35, 15, 10
            w_team, w_reliability, w_maturity = 10, 5, 5
        else:
            # QUALIFYING PACE FORMULA
            # 100% focused on i_delta and base speed
            w_base, w_slope, w_variance, w_conv = 70, 0, 0, 0
            w_team, w_reliability, w_maturity = 20, 5, 5
        
        base_scores = X["base_pace_delta"] * w_base
        slope_scores = X["stint_slope"] * 50.0 * w_slope # Scaling slope (e.g. 0.05 -> 2.5s impact)
        var_scores = X["stint_variance"] * 10.0 * w_variance
        conv_bonus = X["sunday_conversion_factor"] * 50.0 * w_conv # Positive conversion reduces score
        
        latent_scores = (1.0 - X["team_strength"]) * w_team
        reliability_penalties = (1.0 - X["reliability"]) * w_reliability
        maturity_penalties = (1.0 - X["constructor_maturity"]) * w_maturity
        
        final_scores = base_scores + slope_scores + var_scores - conv_bonus + latent_scores + reliability_penalties + maturity_penalties
        
        # Build breakdown for transparency
        breakdown = []
        for i in range(len(X)):
            bd = {
                "base_score": float(base_scores.iloc[i]),
                "slope_score": float(slope_scores.iloc[i]),
                "consistency_score": float(var_scores.iloc[i]),
                "sunday_conversion": float(conv_bonus.iloc[i]),
                "team_score": float(latent_scores.iloc[i]),
                "final_score": float(final_scores.iloc[i]),
                "weights": {
                    "base": w_base,
                    "slope": w_slope,
                    "consistency": w_variance,
                    "conversion": w_conv,
                    "team": w_team
                }
            }
            breakdown.append(bd)
        
        # Predicted Rankings (lower score is better)
        ranks = final_scores.values
        
        # Predicted Deltas (for time estimation)
        predicted_deltas = X["base_pace_delta"].values + (X["stint_slope"].values * 20.0 if session_type == "R" else 0)
        
        return ranks, predicted_deltas, breakdown

if __name__ == "__main__":
    predictor = F1PredictorModel()
    predictor.build_models()
    print("Models ready.")
