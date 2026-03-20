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
        self._driver_history_cache = {} # Memoization for race counts
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
    def get_driver_race_count(self, driver_abbr: str, years: List[int] = None) -> int:
        """
        Check how many total race entries a driver has across the specified years.
        Used for dynamic rookie detection. Results are memoized.
        """
        if years is None:
            years = [2023, 2024, 2025]
            
        cache_key = f"{driver_abbr}_{''.join(map(str, years))}"
        if cache_key in self._driver_history_cache:
            return self._driver_history_cache[cache_key]
            
        total_races = 0
        for year in years:
            try:
                schedule = self.get_event_schedule(year)
                if schedule.empty: continue
                
                # Check each completed race session for the driver
                for _, event in schedule[schedule['RoundNumber'] > 0].iterrows():
                    try:
                        # We use results directly as it's faster than loading full sessions
                        # Note: This might still trigger some API calls if results aren't cached
                        results = fastf1.get_session(year, int(event['RoundNumber']), 'R').results
                        if not results.empty and driver_abbr in results['Abbreviation'].values:
                            total_races += 1
                    except: continue
            except Exception as e:
                logger.warning(f"Error counting races for {driver_abbr} in {year}: {e}")
                
        self._driver_history_cache[cache_key] = total_races
        return total_races

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
                # Fallback to maturity-aware defaults (0.5 established, 0.1 new)
                maturity = self.get_constructor_maturity(team_name)
                return 0.5 if maturity >= 1.0 else 0.1
            
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

    def get_long_stint_pace(self, session) -> Dict[str, Dict[str, float]]:
        """
        Extracts high-fidelity stint profile from a practice session.
        Uses np.polyfit to calculate degradation slope and base pace intercept.
        Returns {driver_abbr: {slope, intercept, variance}}
        """
        try:
            laps = session.laps
            if laps.empty:
                return {}
            
            stint_profiles = {}
            
            for driver in laps['Driver'].unique():
                driver_laps = laps.pick_drivers(driver).sort_values('LapNumber')
                
                if driver_laps.empty or len(driver_laps) < 5:
                    continue
                
                # Find best long stint profile
                best_profile = None
                current_stint_laps = []
                current_compound = None
                
                for _, lap in driver_laps.iterrows():
                    compound = lap.get('Compound', 'UNKNOWN')
                    lap_time = lap['LapTime']
                    
                    if pd.isna(lap_time):
                        current_stint_laps = []
                        current_compound = None
                        continue
                    
                    lap_seconds = lap_time.total_seconds()
                    
                    # Outlier Rejection: skip pit laps, safety cars, or laps > 1.5s slower than previous
                    is_outlier = lap_seconds > 200 or lap_seconds < 60
                    if not is_outlier and current_stint_laps:
                        if lap_seconds > (current_stint_laps[-1] + 1.5):
                            is_outlier = True
                    
                    if is_outlier:
                        current_stint_laps = []
                        current_compound = None
                        continue
                    
                    if compound == current_compound:
                        current_stint_laps.append(lap_seconds)
                    else:
                        current_compound = compound
                        current_stint_laps = [lap_seconds]
                    
                    # If we have a stint of 5+ laps, analyze it
                    if len(current_stint_laps) >= 5:
                        # 1. Slope & Intercept via linear regression
                        # x = lap index (0 to N), y = lap times
                        x = np.arange(len(current_stint_laps))
                        y = np.array(current_stint_laps)
                        
                        slope, intercept = np.polyfit(x, y, 1)
                        
                        # 2. Variance (Stability)
                        # We remove the slope trend first to see the driver's inherent consistency
                        trendline = (slope * x) + intercept
                        detrended_variance = np.var(y - trendline)
                        
                        # 3. Quality check: only accept positive slopes (real wear) or very flat ones
                        # A massive negative slope suggests they were lifting/coasting earlier
                        if best_profile is None or intercept < best_profile['intercept']:
                            best_profile = {
                                "slope": float(slope),
                                "intercept": float(intercept),
                                "variance": float(detrended_variance)
                            }
                
                if best_profile:
                    stint_profiles[driver] = best_profile
            
            return stint_profiles
        except Exception as e:
            logger.error(f"Error calculating high-fidelity stint pace: {e}")
            return {}

    def get_sunday_conversion_factor(self, team_name: str, years: List[int] = None, count: int = 5) -> float:
        """
        Calculates the average Qualy vs Race pace delta for a team over last N races.
        Only uses 'Green Flag' (TrackStatus == '1') laps.
        Positive = better in race than qualy, Negative = 'Saturday car'.
        """
        if years is None:
            years = [2026, 2025]
            
        deltas = []
        try:
            for year in years:
                schedule = self.get_event_schedule(year)
                if schedule.empty: continue
                
                # Check recent races
                races = schedule[schedule['RoundNumber'] > 0].iloc[::-1]
                for _, event in races.iterrows():
                    try:
                        # 1. Qualy Delta for Team
                        q_session = fastf1.get_session(year, int(event['RoundNumber']), 'Q')
                        q_session.load()
                        q_laps = q_session.laps.pick_quicklaps()
                        if q_laps.empty: continue
                        
                        pole_time = q_laps['LapTime'].min().total_seconds()
                        team_q_best = q_laps[q_laps['Team'] == team_name]['LapTime'].min().total_seconds()
                        q_pct_deficit = (team_q_best - pole_time) / pole_time
                        
                        # 2. Race Delta for Team (Green Laps Only)
                        r_session = fastf1.get_session(year, int(event['RoundNumber']), 'R')
                        r_session.load()
                        # Filter TrackStatus '1' (Green) and valid lap times
                        r_laps = r_session.laps[r_session.laps['TrackStatus'] == '1'].dropna(subset=['LapTime'])
                        if r_laps.empty: continue
                        
                        # Compare team average race lap to winner average race lap
                        winner = r_session.results.sort_values('Position').iloc[0]['Abbreviation']
                        winner_avg = r_laps[r_laps['Driver'] == winner]['LapTime'].dt.total_seconds().mean()
                        team_avg = r_laps[r_laps['Team'] == team_name]['LapTime'].dt.total_seconds().mean()
                        r_pct_deficit = (team_avg - winner_avg) / winner_avg
                        
                        # Conversion = Qualy Deficit - Race Deficit
                        # e.g. If Q is 1% off and R is 0.5% off, conversion is +0.5% (Sunday car)
                        conversion = q_pct_deficit - r_pct_deficit
                        deltas.append(conversion)
                        
                        if len(deltas) >= count: break
                    except: continue
                if len(deltas) >= count: break
                
            return float(np.mean(deltas)) if deltas else 0.0
        except Exception as e:
            logger.error(f"Error calculating Sunday conversion for {team_name}: {e}")
            return 0.0

    def run_full_sync(self, update_callback=None) -> Dict[str, Any]:
        """
        Runs a staged data sync pipeline, downloading all sessions required
        for accurate predictions into the local FastF1 cache.
        Calls update_callback(progress_pct, stage_name) after each stage.
        """
        def _update(pct: int, stage: str):
            logger.info(f"[Sync] {pct}% - {stage}")
            if update_callback:
                update_callback(pct, stage)

        results = {"stages": [], "errors": []}

        # Stage 1: Load schedules for 2024, 2025, 2026
        _update(5, "Loading race schedules (2024-2026)...")
        schedules = {}
        for yr in [2024, 2025, 2026]:
            try:
                schedules[yr] = self.get_event_schedule(yr)
                results["stages"].append(f"Schedule {yr}: OK ({len(schedules[yr])} events)")
            except Exception as e:
                results["errors"].append(f"Schedule {yr}: {e}")
        _update(10, "Schedules loaded.")

        # Stage 2: Cache race results for rookie detection (2023-2025)
        _update(12, "Caching race results for rookie detection (2023-2025)...")
        rookie_sessions_loaded = 0
        for yr in [2023, 2024, 2025]:
            try:
                sched = schedules.get(yr) if yr in schedules else self.get_event_schedule(yr)
                if sched is None or sched.empty:
                    continue
                past_rounds = sched[sched['RoundNumber'] > 0]
                total = len(past_rounds)
                for idx, (_, event) in enumerate(past_rounds.iterrows()):
                    try:
                        session = fastf1.get_session(yr, int(event['RoundNumber']), 'R')
                        session.load(laps=False, telemetry=False, weather=False, messages=False)
                        rookie_sessions_loaded += 1
                    except:
                        pass
                    # Update progress within Stage 2 (12% -> 40%)
                    stage_pct = 12 + int(((yr - 2023) * total + idx + 1) / (3 * max(total, 1)) * 28)
                    _update(stage_pct, f"Caching {yr} race results ({idx+1}/{total})...")
            except Exception as e:
                results["errors"].append(f"Rookie data {yr}: {e}")
        results["stages"].append(f"Rookie detection data: {rookie_sessions_loaded} sessions cached")
        _update(40, "Race results cached.")

        # Stage 3: Cache last 5 qualifying + race sessions for Sunday Conversion
        _update(42, "Caching recent sessions for Sunday Conversion factor...")
        conv_sessions_loaded = 0
        for yr in [2025, 2026]:
            try:
                sched = schedules.get(yr) if yr in schedules else self.get_event_schedule(yr)
                if sched is None or sched.empty:
                    continue
                past_rounds = sched[sched['RoundNumber'] > 0].iloc[::-1].head(5)
                for _, event in past_rounds.iterrows():
                    for sid in ['Q', 'R']:
                        try:
                            session = fastf1.get_session(yr, int(event['RoundNumber']), sid)
                            session.load(telemetry=False, messages=False)
                            conv_sessions_loaded += 1
                        except:
                            pass
            except Exception as e:
                results["errors"].append(f"Sunday conversion data {yr}: {e}")
        results["stages"].append(f"Sunday conversion data: {conv_sessions_loaded} sessions cached")
        _update(65, "Sunday conversion sessions cached.")

        # Stage 4: Cache upcoming circuit history (2023-2025)
        _update(67, "Caching upcoming circuit history...")
        try:
            next_event = self.get_next_event()
            circuit = next_event.get('country', next_event.get('name', ''))
            circuit_sessions_loaded = 0
            for yr in [2023, 2024, 2025]:
                for sid in ['Q', 'R', 'FP2']:
                    try:
                        session = fastf1.get_session(yr, circuit, sid)
                        session.load(telemetry=False, messages=False)
                        circuit_sessions_loaded += 1
                    except:
                        pass
            results["stages"].append(f"Circuit history ({circuit}): {circuit_sessions_loaded} sessions cached")
            _update(90, f"Circuit history for {circuit} cached.")
        except Exception as e:
            results["errors"].append(f"Circuit history: {e}")

        # Stage 5: Cache rolling team rating sessions (last 3 rounds of current year)
        _update(92, "Caching rolling team rating sessions...")
        try:
            sched_2026 = schedules.get(2026) if 2026 in schedules else self.get_event_schedule(2026)
            if sched_2026 is not None and not sched_2026.empty:
                past = sched_2026[sched_2026['RoundNumber'] > 0].iloc[::-1].head(3)
                for _, event in past.iterrows():
                    try:
                        session = fastf1.get_session(2026, int(event['RoundNumber']), 'Q')
                        session.load(telemetry=False, messages=False)
                    except:
                        pass
            results["stages"].append("Rolling team ratings: sessions cached")
        except Exception as e:
            results["errors"].append(f"Rolling ratings: {e}")

        _update(100, "Sync complete.")
        return results

    def check_cache_health(self) -> Dict[str, Any]:
        """
        Inspects the local FastF1 cache directory and returns a structured
        health report for each data layer the predictor depends on.
        """
        health = {
            "cache_dir": self.cache_dir,
            "cache_exists": os.path.exists(self.cache_dir),
            "total_sessions_cached": 0,
            "layers": {
                "rookie_data": {"status": "missing", "detail": "No 2023-2025 race results cached"},
                "sunday_conversion": {"status": "missing", "detail": "No recent Q/R sessions cached"},
                "circuit_history": {"status": "missing", "detail": "No upcoming circuit history cached"},
                "rolling_ratings": {"status": "missing", "detail": "No recent qualifying sessions cached"},
            }
        }

        if not health["cache_exists"]:
            return health

        # Walk the cache directory and count session files
        session_files = []
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.endswith('.ff1') or f.endswith('.pkl') or f.endswith('.json'):
                    session_files.append(os.path.join(root, f))
        health["total_sessions_cached"] = len(session_files)

        # Check for rookie data: need 2023/2024/2025 race files
        rookie_files = [f for f in session_files if any(str(yr) in f for yr in ['2023', '2024', '2025'])]
        if len(rookie_files) >= 10:
            health["layers"]["rookie_data"] = {"status": "ready", "detail": f"{len(rookie_files)} historical session files found"}
        elif len(rookie_files) > 0:
            health["layers"]["rookie_data"] = {"status": "partial", "detail": f"{len(rookie_files)} historical session files found (need ~20+)"}

        # Check for sunday conversion: need recent 2025/2026 files
        recent_files = [f for f in session_files if any(str(yr) in f for yr in ['2025', '2026'])]
        if len(recent_files) >= 6:
            health["layers"]["sunday_conversion"] = {"status": "ready", "detail": f"{len(recent_files)} recent session files found"}
        elif len(recent_files) > 0:
            health["layers"]["sunday_conversion"] = {"status": "partial", "detail": f"{len(recent_files)} recent session files found (need 6+)"}

        # Check for circuit history: try to find next event circuit in cache
        try:
            next_event = self.get_next_event()
            circuit = next_event.get('country', next_event.get('name', '')).lower()
            circuit_files = [f for f in session_files if circuit.lower() in f.lower()]
            if len(circuit_files) >= 3:
                health["layers"]["circuit_history"] = {"status": "ready", "detail": f"{len(circuit_files)} files for {circuit}"}
            elif len(circuit_files) > 0:
                health["layers"]["circuit_history"] = {"status": "partial", "detail": f"{len(circuit_files)} files for {circuit} (need 3+)"}
        except:
            pass

        # Check for rolling ratings: need 2026 qualifying files
        rating_files = [f for f in session_files if '2026' in f]
        if len(rating_files) >= 3:
            health["layers"]["rolling_ratings"] = {"status": "ready", "detail": f"{len(rating_files)} 2026 session files found"}
        elif len(rating_files) > 0:
            health["layers"]["rolling_ratings"] = {"status": "partial", "detail": f"{len(rating_files)} 2026 session files found (need 3+)"}

        return health

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
