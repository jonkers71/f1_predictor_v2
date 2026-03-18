import fastf1
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure FastF1 caching
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

class F1DataEngine:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        logger.info(f"F1DataEngine initialized with cache: {self.cache_dir}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_session(self, year: int, gp: Any, identifier: str) -> fastf1.core.Session:
        """
        Loads a session with retry logic.
        gp can be round number (int) or circuit name (str).
        identifier can be 'FP1', 'FP2', 'FP3', 'Q', 'S', 'SQ', 'R'.
        """
        try:
            session = fastf1.get_session(year, gp, identifier)
            session.load()
            return session
        except Exception as e:
            logger.error(f"Error loading session {year} {gp} {identifier}: {e}")
            raise

    def get_event_schedule(self, year: int) -> pd.DataFrame:
        """Fetches the full schedule for a given year."""
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule
        except Exception as e:
            logger.error(f"Error fetching schedule for {year}: {e}")
            return pd.DataFrame()

    def is_sprint_weekend(self, year: int, round_num: int) -> bool:
        """Determines if a weekend is a sprint weekend."""
        try:
            event = fastf1.get_event(year, round_num)
            # Check for Sprint sessions
            sessions = [event.Session1, event.Session2, event.Session3, event.Session4, event.Session5]
            return any('Sprint' in str(s) for s in sessions)
        except Exception as e:
            logger.error(f"Error checking sprint weekend {year} {round_num}: {e}")
            return False

    def get_weather_summary(self, session: fastf1.core.Session) -> Dict[str, float]:
        """Extracts median weather data from a session."""
        try:
            weather = session.get_weather_data()
            if weather.empty:
                return {}
            return {
                "air_temp": weather["AirTemp"].median(),
                "track_temp": weather["TrackTemp"].median(),
                "humidity": weather["Humidity"].median(),
                "rainfall": float(weather["Rainfall"].any()),
                "pressure": weather["Pressure"].median()
            }
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {}

    def get_driver_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Gets processed driver results from a session."""
        try:
            results = session.results
            if results.empty:
                return pd.DataFrame()
                
            filtered_results = results[[
                'DriverNumber', 'FullName', 'Abbreviation', 'TeamName', 
                'Position', 'ClassifiedPosition', 'GridPosition', 'Status'
            ]]
            return filtered_results
        except Exception as e:
            logger.error(f"Error getting driver results: {e}")
            return pd.DataFrame()

    def get_best_laps(self, session: fastf1.core.Session) -> pd.DataFrame:
        """Gets the best lap time for each driver in a session."""
        try:
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                return pd.DataFrame()
            
            best_laps = laps.groupby('Driver')['LapTime'].min().reset_index()
            # Convert LapTime (timedelta) to seconds
            best_laps['LapTimeSeconds'] = best_laps['LapTime'].dt.total_seconds()
            return best_laps
        except Exception as e:
            logger.error(f"Error getting best laps: {e}")
            return pd.DataFrame()

    def get_ideal_laps(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Calculates the 'Ideal Lap' (sum of best sectors) for each driver.
        This is often a better predictor of raw pace than Best Lap.
        """
        try:
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                return pd.DataFrame()

            # Group by driver and get min for each sector
            s1 = laps.groupby('Driver')['Sector1Time'].min().dt.total_seconds()
            s2 = laps.groupby('Driver')['Sector2Time'].min().dt.total_seconds()
            s3 = laps.groupby('Driver')['Sector3Time'].min().dt.total_seconds()
            
            ideal = (s1 + s2 + s3).reset_index()
            ideal.columns = ['Driver', 'IdealLapSeconds']
            return ideal
        except Exception as e:
            logger.error(f"Error calculating ideal laps: {e}")
            return pd.DataFrame()

    def get_lap_counts(self, session: fastf1.core.Session) -> Dict[str, int]:
        """Counts the total quick laps per driver in a session."""
        try:
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                return {}
            return laps.groupby('Driver').size().to_dict()
        except Exception as e:
            logger.error(f"Error getting lap counts: {e}")
            return {}

    def get_lap_consistency(self, session: fastf1.core.Session) -> Dict[str, float]:
        """
        Calculates lap time standard deviation for each driver.
        Lower = more consistent / 'locked in'.
        """
        try:
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                return {}
            # Standard deviation of lap times in seconds
            stdevs = laps.groupby('Driver')['LapTime'].apply(lambda x: x.dt.total_seconds().std())
            return stdevs.fillna(1.0).to_dict() # 1.0s fallback for low lap counts
        except Exception as e:
            logger.error(f"Error getting consistency: {e}")
            return {}

    def get_rolling_team_rating(self, team_name: str, year: int, current_round: int, window: int = 3) -> float:
        """
        Calculates a dynamic team rating based on the average qualifying pace deficit
        over the last `window` races.
        0.0% deficit = 1.0 rating
        1.0% deficit = 0.7 rating
        2.5% deficit = 0.2 rating
        """
        try:
            deficits = []
            # Look back through previous rounds
            for r in range(current_round - 1, max(0, current_round - window - 1), -1):
                try:
                    # Try to get qualifying session
                    # If current_round is 1, it will look at last year's late rounds
                    target_year = year
                    target_round = r
                    if r <= 0:
                        target_year = year - 1
                        # Find max round of previous year
                        prev_schedule = self.get_event_schedule(target_year)
                        target_round = int(prev_schedule['RoundNumber'].max()) + r
                    
                    session = self.get_session(target_year, target_round, 'Q')
                    laps = session.laps.pick_quicklaps()
                    if laps.empty:
                        continue
                    
                    pole_time = laps['LapTime'].min().total_seconds()
                    team_laps = laps[laps['Team'] == team_name]
                    if team_laps.empty:
                        # Fallback: maybe teammate abbreviation check if team name differs
                        continue
                    
                    team_best = team_laps['LapTime'].min().total_seconds()
                    
                    # Percentage deficit
                    pct_deficit = ((team_best - pole_time) / pole_time) * 100.0
                    deficits.append(pct_deficit)
                except:
                    continue
            
            if not deficits:
                # Fallback to static if no recent data
                return self.get_team_rating(team_name)
            
            avg_deficit = np.mean(deficits)
            
            # Mapping: 0.0 -> 1.0, 1.0 -> 0.7, 2.5 -> 0.2
            # Linear interpolation or smooth curve
            # formula: rating = max(0, 1.0 - (avg_deficit / 3.0)) 
            # But let's use the user's specific points:
            if avg_deficit <= 0.05: return 1.0
            if avg_deficit <= 1.0:
                # Interpolate 1.0 -> 0.7
                return 1.0 - (avg_deficit * 0.3)
            # Interpolate 1.0(0.7) -> 2.5(0.2)
            rating = 0.7 - ((avg_deficit - 1.0) * (0.5 / 1.5))
            return float(max(0.05, rating))
            
        except Exception as e:
            logger.error(f"Error calculating rolling rating for {team_name}: {e}")
            return self.get_team_rating(team_name)

    def get_constructor_maturity(self, team_name: str) -> float:
        """
        Returns a maturity index (0-1).
        Brand new teams start at 0.15 and grow as they complete races.
        """
        # In a real scenario, we'd query historical records.
        # For 2026 simulation:
        new_teams = {
            "Cadillac": 0,
            "Audi": 0,
            "Andretti": 0
        }
        
        # Hardcoded for the 2026 scenario start
        if team_name in new_teams:
            return 0.15
        return 1.0

    def get_team_rating(self, team_name: str) -> float:
        """
        Baseline ratings for the start of 2026.
        """
        ratings = {
            "Mercedes": 0.98,
            "Ferrari": 0.95,
            "Haas F1 Team": 0.70,
            "Red Bull Racing": 0.60,
            "McLaren": 0.55,
            "Aston Martin": 0.45,
            "RB": 0.40,
            "Racing Bulls": 0.40,
            "Alpine": 0.30,
            "Sauber": 0.25,
            "Kick Sauber": 0.25,
            "Audi": 0.20,
            "Cadillac": 0.15
        }
        return ratings.get(team_name, 0.10)

    def format_lap_time(self, seconds: float) -> str:
        """Formats seconds into mm:ss.xxx string."""
        if pd.isna(seconds):
            return "N/A"
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes}:{rem_seconds:06.3f}"

    def get_current_grid(self, year: int) -> List[Dict[str, str]]:
        """
        Dynamically retrieves the current driver lineup for a given year.
        Walks backwards through completed rounds to find the most recent session with results.
        """
        try:
            schedule = self.get_event_schedule(year)
            if schedule.empty:
                return []
            
            events = schedule[schedule['RoundNumber'] > 0]
            now = datetime.now()
            
            # Walk backwards from the most recent past event
            for _, event in events.iloc[::-1].iterrows():
                try:
                    event_date = event['EventDate']
                    if hasattr(event_date, 'timestamp') and event_date > now:
                        continue
                    
                    # Try qualifying first, then race
                    for sid in ['Q', 'R', 'FP1']:
                        try:
                            session = self.get_session(year, int(event['RoundNumber']), sid)
                            results = self.get_driver_results(session)
                            if not results.empty:
                                grid = []
                                for _, row in results.iterrows():
                                    grid.append({
                                        "Abbreviation": row['Abbreviation'],
                                        "TeamName": row['TeamName'],
                                        "FullName": row['FullName']
                                    })
                                logger.info(f"Got {year} grid from R{int(event['RoundNumber'])} {sid}: {len(grid)} drivers")
                                return grid
                        except:
                            continue
                except:
                    continue
            
            # If no completed events yet, try loading the first event's entry list
            try:
                first_round = int(events.iloc[0]['RoundNumber'])
                session = self.get_session(year, first_round, 'FP1')
                results = self.get_driver_results(session)
                if not results.empty:
                    return [{"Abbreviation": r['Abbreviation'], "TeamName": r['TeamName'], "FullName": r['FullName']} 
                            for _, r in results.iterrows()]
            except:
                pass
            
            return []
        except Exception as e:
            logger.error(f"Error getting current grid for {year}: {e}")
            return []

    def get_circuit_history(self, circuit_name: str, session_type: str = "Q", 
                           years: List[int] = None) -> Dict[str, Any]:
        """
        Gets historical performance data at a specific circuit across multiple years.
        Returns average deltas per driver (normalized to pole) and metadata about what was found.
        """
        if years is None:
            years = [2025, 2024, 2023]
        
        all_deltas = {}  # driver_abbr -> list of deltas across years
        sessions_found = []
        
        for yr in years:
            try:
                # FastF1 accepts circuit name strings (e.g. "Japan", "Bahrain")
                session = self.get_session(yr, circuit_name, session_type)
                ideal_laps = self.get_ideal_laps(session)
                
                if ideal_laps.empty:
                    continue
                
                # Normalize to the fastest (pole) time
                min_time = ideal_laps['IdealLapSeconds'].min()
                
                for _, row in ideal_laps.iterrows():
                    drv = row['Driver']
                    delta = row['IdealLapSeconds'] - min_time
                    if drv not in all_deltas:
                        all_deltas[drv] = []
                    all_deltas[drv].append(delta)
                
                event_name = session.event['EventName']
                sessions_found.append(f"{yr} {event_name} {session_type}")
                logger.info(f"Circuit history: loaded {yr} {circuit_name} {session_type}")
            except Exception as e:
                logger.warning(f"Circuit history: {yr} {circuit_name} {session_type} not available: {e}")
                continue
        
        # Average deltas across years
        avg_deltas = {}
        for drv, deltas in all_deltas.items():
            avg_deltas[drv] = float(np.mean(deltas))
        
        return {
            "deltas": avg_deltas,
            "sessions_found": sessions_found,
            "circuit": circuit_name,
            "years_searched": years,
            "drivers_found": len(avg_deltas)
        }

    def get_long_stint_pace(self, session) -> Dict[str, float]:
        """
        Extracts long stint average pace from a practice session.
        A 'long stint' is 5+ consecutive laps on the same compound.
        Returns {driver_abbr: avg_lap_time_seconds} for race pace estimation.
        """
        try:
            laps = session.laps
            if laps.empty:
                return {}
            
            long_stint_paces = {}
            
            for driver in laps['Driver'].unique():
                driver_laps = laps.pick_drivers(driver).sort_values('LapNumber')
                
                if driver_laps.empty or len(driver_laps) < 5:
                    continue
                
                # Find consecutive laps on the same compound
                best_stint_pace = None
                current_stint = []
                current_compound = None
                
                for _, lap in driver_laps.iterrows():
                    compound = lap.get('Compound', 'UNKNOWN')
                    lap_time = lap['LapTime']
                    
                    if pd.isna(lap_time):
                        current_stint = []
                        current_compound = None
                        continue
                    
                    lap_seconds = lap_time.total_seconds()
                    
                    # Skip obvious outliers (pit laps, safety car, etc.)
                    if lap_seconds > 200 or lap_seconds < 60:
                        current_stint = []
                        current_compound = None
                        continue
                    
                    if compound == current_compound:
                        current_stint.append(lap_seconds)
                    else:
                        current_compound = compound
                        current_stint = [lap_seconds]
                    
                    # If we have a stint of 5+ laps, calculate average
                    if len(current_stint) >= 5:
                        # Drop the first lap (outlap/adjustment) and calculate
                        stint_pace = np.mean(current_stint[1:])
                        if best_stint_pace is None or stint_pace < best_stint_pace:
                            best_stint_pace = stint_pace
                
                if best_stint_pace is not None:
                    long_stint_paces[driver] = float(best_stint_pace)
            
            return long_stint_paces
        except Exception as e:
            logger.error(f"Error extracting long stint pace: {e}")
            return {}

    def get_next_event(self) -> Dict[str, Any]:
        """Identifies the next upcoming event from the 2026 schedule."""
        try:
            now = datetime.now()
            schedule = fastf1.get_event_schedule(2026)
            upcoming = schedule[schedule['EventDate'] >= now].iloc[0]
            
            return {
                "round": int(upcoming['RoundNumber']),
                "name": upcoming['EventName'],
                "location": upcoming['Location'],
                "country": upcoming.get('Country', upcoming['EventName']),
                "date": upcoming['EventDate'].strftime('%Y-%m-%d'),
                "year": 2026
            }
        except Exception as e:
            logger.error(f"Error getting next event: {e}")
            return {"round": 1, "name": "Bahrain Grand Prix", "year": 2026, "country": "Bahrain"}

if __name__ == "__main__":
    # Test script
    engine = F1DataEngine()
    # Test with 2024 China (Sprint Weekend)
    if engine.is_sprint_weekend(2024, 5):
        print("China 2024 is a Sprint Weekend")
    
    # Load 2024 China Qualifying
    try:
        q_session = engine.get_session(2024, 5, 'Q')
        print(f"Loaded {q_session.event['EventName']} Qualifying")
        weather = engine.get_weather_summary(q_session)
        print(f"Weather: {weather}")
        
        best_laps = engine.get_best_laps(q_session)
        for _, row in best_laps.head().iterrows():
            print(f"Driver {row['Driver']}: {engine.format_lap_time(row['LapTimeSeconds'])}")
    except Exception as e:
        print(f"Test failed: {e}")
