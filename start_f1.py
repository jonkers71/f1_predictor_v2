import subprocess
import os
import sys
import time
import webbrowser

def run_app():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("🚀 Starting F1 Predictor v2...")

    # 1. Start Backend
    print("📡 Launching Backend (FastAPI)...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=backend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    # 2. Wait a moment for backend to warm up
    time.sleep(2)

    # 3. Start Frontend
    print("🌐 Launching Frontend (Next.js)...")
    frontend_process = subprocess.Popen(
        ["npm.cmd", "run", "dev"], 
        cwd=frontend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    print("\n✅ Both systems are starting!")
    print("👉 Backend API: http://localhost:8000")
    print("👉 Frontend UI: http://localhost:3000")
    
    time.sleep(5)
    webbrowser.open("http://localhost:3000")

    print("\nClose the new terminal windows to stop the servers.")

if __name__ == "__main__":
    run_app()
