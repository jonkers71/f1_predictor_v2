from core.data_engine import F1DataEngine
from core.models import F1PredictorModel
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    engine = F1DataEngine()
    model = F1PredictorModel()
    model.build_models()

    all_features = []
    all_ranks = []
    all_deltas = []
    groups = []

    # Train on recent years (2022-2023) for initial bootstrap
    for year in [2022, 2023]:
        logger.info(f"Fetching training data for {year}...")
        schedule = engine.get_event_schedule(year)
        # Filter for completed races
        # FastF1 schedule cleanup might be needed, using subset for demo
        for idx, event in schedule.head(5).iterrows(): # Limit for speed in demo
            gp = event['RoundNumber']
            try:
                # Get Qualifying data
                q_session = engine.get_session(year, gp, 'Q')
                laps = engine.get_best_laps(q_session)
                results = engine.get_driver_results(q_session)
                weather = engine.get_weather_summary(q_session)

                if laps.empty or results.empty:
                    continue

                # Merge laps and results
                df = pd.merge(laps, results, left_on='Driver', right_on='Abbreviation')
                
                # Mock feature engineering for now
                # In production, this would be much more robust
                df['avg_practice_delta'] = df['LapTimeSeconds'] - df['LapTimeSeconds'].min()
                df['practice_consistency'] = np.random.uniform(0.1, 0.4, size=len(df))
                df['track_temp'] = weather.get('track_temp', 35)
                df['air_temp'] = weather.get('air_temp', 25)
                df['is_sprint'] = 0
                df['tyre_life'] = 1
                df['team_momentum'] = 0.5

                # Training targets
                # Ranker: Actual position
                # Regressor: Delta to pole
                y_rank = df['Position']
                y_delta = df['LapTimeSeconds'] - df['LapTimeSeconds'].min()

                # Feature selection matching model.feature_names
                X = df[[
                    "avg_practice_delta", "practice_consistency", "track_temp", 
                    "air_temp", "is_sprint", "tyre_life", "team_momentum"
                ]]

                all_features.append(X)
                all_ranks.append(y_rank)
                all_deltas.append(y_delta)
                groups.append(len(df))

            except Exception as e:
                logger.warning(f"Skipping {year} Round {gp} due to error: {e}")

    if all_features:
        final_X = pd.concat(all_features)
        final_y_rank = pd.concat(all_ranks)
        final_y_delta = pd.concat(all_deltas)
        
        # Train
        model.train_ranker(final_X, final_y_rank, groups)
        model.train_regressor(final_X, final_y_delta)
        logger.info("Training complete and models saved.")
    else:
        logger.error("No data collected for training.")

if __name__ == "__main__":
    train()
