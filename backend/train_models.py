import asyncio
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.core.data_engine import F1DataEngine
from backend.core.models import F1PredictorModel
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train():
    model = F1PredictorModel()
    model.build_models()
    
    logger.info("Generating synthetic training data to bypass API rate limits...")
    
    # Create 1000 synthetic samples
    rows = 1000
    data = []
    
    # Features: ["ideal_lap_delta", "base_pace_delta", "stint_slope", "stint_variance",
    #            "team_strength", "reliability", "constructor_maturity", "is_rookie",
    #            "sunday_conversion_factor", "track_temp", "is_sprint", "has_long_stint"]
    
    for _ in range(rows):
        i_delta = np.random.uniform(0, 2.0)
        base_delta = i_delta + np.random.uniform(-0.2, 0.5)
        slope = np.random.uniform(0, 0.15) # 0 to 0.15s per lap
        variance = np.random.uniform(0, 0.5)
        team_str = np.random.uniform(0.1, 1.0)
        reli = np.random.uniform(0.5, 1.0)
        maturity = np.random.choice([0.15, 1.0], p=[0.1, 0.9])
        is_rookie = np.random.choice([0, 1], p=[0.8, 0.2])
        sunday_conv = np.random.uniform(-0.02, 0.02)
        
        feat = {
            "ideal_lap_delta": float(i_delta),
            "base_pace_delta": float(base_delta),
            "stint_slope": float(slope),
            "stint_variance": float(variance),
            "team_strength": float(team_str),
            "reliability": float(reli * maturity),
            "constructor_maturity": float(maturity),
            "is_rookie": is_rookie,
            "sunday_conversion_factor": float(sunday_conv),
            "track_temp": 30.0,
            "is_sprint": 0,
            "has_long_stint": 1
        }
        data.append(feat)
        
    X_train = pd.DataFrame(data)
    
    # Synthetic Labels
    # For Ranker: lower score is better rank
    # Rank should be roughly proportional to: 
    # pace + (slope * 20) + (1.0-team) + (1.0-reli) + rookie + (1.0-maturity) - (sunday_conv * 10)
    scores = (X_train["base_pace_delta"] * 20 + 
              X_train["stint_slope"] * 100 + 
              X_train["stint_variance"] * 10 +
              (1.0 - X_train["team_strength"]) * 10 + 
              (1.0 - X_train["reliability"]) * 5 + 
              X_train["is_rookie"] * 2 + 
              (1.0 - X_train["constructor_maturity"]) * 15 - 
              X_train["sunday_conversion_factor"] * 20)
    
    # Map scores to ranks within groups of 20
    y_rank = []
    groups = []
    for i in range(0, rows, 20):
        group_scores = scores[i:i+20]
        # Rank from 1 to 20 based on score
        rank_indices = np.argsort(group_scores)
        ranks = np.zeros(20)
        for r_idx, s_idx in enumerate(rank_indices):
            ranks[s_idx] = r_idx + 1
        y_rank.extend(ranks)
        groups.append(20)
        
    y_rank = np.array(y_rank)
    y_delta = X_train["base_pace_delta"].values + (X_train["stint_slope"].values * 10)
    
    group_counts = np.array(groups)
    
    # Train
    model.train_ranker(X_train, y_rank, group_counts)
    model.train_regressor(X_train, y_delta)
    
    logger.info("Synthetic training complete. Models saved with new feature schema.")

if __name__ == "__main__":
    asyncio.run(train())
