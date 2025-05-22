import subprocess
import logging
import sys
from datetime import datetime
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Run a Python script and log its output."""
    logger.info(f"Starting {description}...")
    try:
        # If the script is in the scripts/ directory, run as a module for import compatibility
        if script_name.startswith('scripts/') and script_name.endswith('.py'):
            module_name = script_name[:-3].replace('/', '.')
            cmd = ['python3', '-m', module_name]
        else:
            cmd = ['python3', script_name]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully completed {description}")
        if result.stdout:
            logger.info(f"Output from {script_name}:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}:")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {str(e)}")
        return False

def update_data():
    """Run all data update scripts in sequence."""
    start_time = datetime.now()
    logger.info(f"Starting data update at {start_time}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if FanGraphs projections were downloaded in the last 12 hours
    batters_path = Path('projections/fangraphs-leaderboard-projections-batters.csv')
    pitchers_path = Path('projections/fangraphs-leaderboard-projections-pitchers.csv')
    now = datetime.now().timestamp()
    skip_fangraphs = False
    if batters_path.exists() and pitchers_path.exists():
        batters_age = now - batters_path.stat().st_mtime
        pitchers_age = now - pitchers_path.stat().st_mtime
        if batters_age < 12 * 3600 and pitchers_age < 12 * 3600:
            skip_fangraphs = True
            logger.info("FanGraphs projections are less than 12 hours old. Skipping download.")
    
    # Define the sequence of scripts to run
    scripts = [
        ('src/fetch_espn_stats.py', 'ESPN stats fetch'),
        ('download_projections.py', 'FanGraphs projections download'),
        ('baseball_stats.py', 'Baseball Reference and Statcast data fetch'),
        ('src/merge_xwoba.py', 'xwOBA data merge'),
        ('src/integrate_projections.py', 'Projections integration'),
        ('scripts/calculate_team_stats.py', 'Team stats calculation'),
        ('scripts/calculate_hitter_impacts.py', 'Hitter impact calculations'),
        ('scripts/calculate_incremental_points.py', 'Pitcher impact calculations')
    ]
    
    # Run each script in sequence
    for script, description in scripts:
        if not run_script(script, description):
            logger.error(f"Failed to complete {description}. Stopping update process.")
            return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Data update completed successfully in {duration}")
    return True

if __name__ == "__main__":
    success = update_data()
    sys.exit(0 if success else 1) 