import pandas as pd
import logging
from pathlib import Path
# Use absolute import for compatibility when running as a script or module
from scripts.roster_utils import select_optimal_roster
from sqlalchemy import create_engine
import os
from sqlalchemy import text
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All stat columns are now lowercase (e.g., 'proj_r', not 'proj_R')

load_dotenv()

def calculate_team_stats():
    """Calculate team stats by aggregating individual player projections."""
    # Connect to the database
    engine = create_engine(os.environ['DATABASE_URL'])
    
    # Get all hitters with their projections
    logger.info("Fetching hitter projections...")
    hitting_query = """
        SELECT 
            name,
            team,
            eligible_positions,
            player_id,
            proj_r,
            proj_hr,
            proj_tb,
            proj_sbn,
            proj_avg,
            proj_ops,
            proj_ab,
            proj_pr
        FROM hitters 
        WHERE team != 'Free Agent'
    """
    all_hitters = pd.read_sql(hitting_query, engine)
    
    # Get all pitchers with their projections
    logger.info("Fetching pitcher projections...")
    pitching_query = """
        SELECT 
            name,
            team,
            proj_k,
            proj_qs,
            proj_era,
            proj_whip,
            proj_k_bb,
            proj_svhd,
            proj_ip,
            proj_bb
        FROM pitchers 
        WHERE team != 'Free Agent'
    """
    all_pitchers = pd.read_sql(pitching_query, engine)
    
    # Get unique teams
    teams = pd.concat([all_hitters['team'], all_pitchers['team']]).unique()
    
    # Calculate optimal roster for each team
    team_stats = []
    for team in teams:
        logger.info(f"\nProcessing team: {team}")
        
        # Get optimal hitting roster
        team_hitters = all_hitters[all_hitters['team'] == team]
        optimal_hitters = select_optimal_roster(team_hitters)
        
        # Get all pitchers for the team
        team_pitchers = all_pitchers[all_pitchers['team'] == team]
        
        if not optimal_hitters.empty:
            # Calculate hitting totals
            hitting_totals = {
                'team_name': team,
                'proj_r': optimal_hitters['proj_r'].sum(),
                'proj_hr': optimal_hitters['proj_hr'].sum(),
                'proj_tb': optimal_hitters['proj_tb'].sum(),
                'proj_sbn': optimal_hitters['proj_sbn'].sum(),
                'proj_ab': optimal_hitters['proj_ab'].sum(),
                'proj_avg': (optimal_hitters['proj_avg'] * optimal_hitters['proj_ab']).sum() / optimal_hitters['proj_ab'].sum(),
                'proj_ops': (optimal_hitters['proj_ops'] * optimal_hitters['proj_ab']).sum() / optimal_hitters['proj_ab'].sum()
            }
            
            # Calculate pitching totals
            pitching_totals = {
                'proj_k': team_pitchers['proj_k'].sum(),
                'proj_qs': team_pitchers['proj_qs'].sum(),
                'proj_svhd': team_pitchers['proj_svhd'].sum(),
                'proj_ip': team_pitchers['proj_ip'].sum(),
                'proj_era': (team_pitchers['proj_era'] * team_pitchers['proj_ip']).sum() / team_pitchers['proj_ip'].sum() if team_pitchers['proj_ip'].sum() > 0 else 0,
                'proj_whip': (team_pitchers['proj_whip'] * team_pitchers['proj_ip']).sum() / team_pitchers['proj_ip'].sum() if team_pitchers['proj_ip'].sum() > 0 else 0,
                'proj_k_bb': team_pitchers['proj_k'].sum() / team_pitchers['proj_bb'].sum() if team_pitchers['proj_bb'].sum() > 0 else 0
            }
            
            # Combine hitting and pitching totals
            team_total = {**hitting_totals, **pitching_totals}
            team_stats.append(team_total)
    
    # Convert to DataFrame
    team_stats_df = pd.DataFrame(team_stats)
    
    # Ensure all required columns exist in the DataFrame before saving to SQL
    required_columns = [
        'k', 'qs', 'svhd', 'ip', 'era', 'whip', 'k_bb',
        'k_points', 'qs_points', 'svhd_points', 'era_points', 'whip_points', 'k_bb_points', 'total_points'
    ]
    for col in required_columns:
        if col not in team_stats_df.columns:
            team_stats_df[col] = None

    # Save to database
    team_stats_df.to_sql('team_stats', engine, if_exists='replace', index=False)
    logger.info("Successfully saved team stats to database")

if __name__ == "__main__":
    calculate_team_stats() 