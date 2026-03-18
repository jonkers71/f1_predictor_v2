import asyncio
import json
import os
import sys

# Ensure backend and core are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "backend"))

from backend.main import get_current_prediction

async def test_prediction():
    print("Running 2026 Japan GP Prediction Verification...")
    try:
        # We need to mock the GP to be Japan if it's not the next one
        # But get_next_event should handle it if it's the current period.
        # Let's just run it as-is first.
        result = await get_current_prediction(session_type="Q")
        
        print(f"\nEvent: {result['event']} (Round {result['round']})")
        print(f"Baseline: {result['baseline']}")
        print("\nTop 5 Predictions:")
        for pred in result['predictions'][:5]:
            print(f"{pred['rank']}. {pred['driver']} ({pred['team']}) - {pred['time']}")
            
        print("\nBottom 3 Predictions:")
        for pred in result['predictions'][-3:]:
            print(f"{pred['rank']}. {pred['driver']} ({pred['team']}) - {pred['time']}")
            
        print("\nData Source Notes:")
        for note in result['data_sources']['notes']:
            print(f"- {note}")
            
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_prediction())
