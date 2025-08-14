import subprocess
import sys
import os


def run_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_path = os.path.join("src", "dashboard", "dashboard.py")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                dashboard_path,
                "--server.port",
                "8501",
                "--server.address",
                "0.0.0.0",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")


if __name__ == "__main__":
    run_dashboard()
