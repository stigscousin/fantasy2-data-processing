import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
from .roster_utils import select_optimal_roster
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def calculate_points(df, return_category_points=False):
    """Calculate points for each team based on their stats."""
    categories = {
        'r': 1,  # Higher is better
        'hr': 1,  # Higher is better
        'tb': 1,  # Higher is better
        'sbn': 1,  # Higher is better
        'avg': 1,  # Higher is better
        'ops': 1,  # Higher is better
    }
    
    points = {}
    category_points = {cat: {} for cat in categories}
    
    for category, direction in categories.items():
        # Sort values in descending order (highest first)
        sorted_values = df.sort_values(by=category, ascending=False)
        
        # Get total number of teams
        num_teams = len(sorted_values)
        
        # Group by value to handle ties
        value_groups = {}
        for i, (_, row) in enumerate(sorted_values.iterrows()):
            value = row[category]
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(row['team_name'])
        
        # Calculate points for each group
        current_rank = 1
        for value, teams in value_groups.items():
            # Calculate median rank for this group
            group_size = len(teams)
            median_rank = current_rank + (group_size - 1) / 2
            # Assign points based on median rank (N points for 1st, N-1 for 2nd, etc.)
            points_for_rank = num_teams - median_rank + 1
            
            for team in teams:
                category_points[category][team] = points_for_rank
                points[team] = points.get(team, 0) + points_for_rank
            
            current_rank += group_size
    
    if return_category_points:
        return points, category_points
    return points

def main():
    db_url = os.environ['DATABASE_URL']
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    engine = create_engine(db_url)
    # Drop and recreate hitter_impacts table
    conn = engine.connect()
    conn.execute(text("DROP TABLE IF EXISTS hitter_impacts"))
    conn.execute(text('''
        CREATE TABLE hitter_impacts (
            player_id INTEGER,
            team_name TEXT,
            impact REAL,
            stat_type TEXT DEFAULT 'projected',
            PRIMARY KEY (player_id, team_name, stat_type)
        )
    '''))
    
    # Get all team names
    cursor = conn.execute(text("SELECT DISTINCT team_name FROM team_stats"))
    all_team_names = [row[0] for row in cursor.fetchall()]
    
    # Get all teams' projected stats
    logger.info("Executing query for team projected stats")
    cursor = conn.execute(text("""
        SELECT team_name,
               proj_r as r,
               proj_hr as hr,
               proj_tb as tb,
               proj_sbn as sbn,
               proj_avg as avg,
               proj_ops as ops
        FROM team_stats
    """))
    all_teams_df = pd.DataFrame(cursor.fetchall(), 
                               columns=['team_name', 'r', 'hr', 'tb', 'sbn', 'avg', 'ops'])
    
    # Calculate points for each category
    points, category_points = calculate_points(all_teams_df, return_category_points=True)
    
    # Calculate total points for each team
    team_totals = {}
    for team in all_teams_df['team_name']:
        total = sum(category_points[category][team] for category in ['r', 'hr', 'tb', 'sbn', 'avg', 'ops'])
        team_totals[team] = total
    
    # For each team, calculate impact for each free agent
    for team_name in all_team_names:
        # Fetch all hitters on the team (with eligible_positions and proj_PR)
        cursor = conn.execute(text("""
            SELECT player_id, name, proj_r, proj_hr, proj_tb, proj_sbn, proj_ab, proj_avg, proj_ops, eligible_positions, proj_pr
            FROM hitters
            WHERE team = :team_name
        """), {"team_name": team_name})
        team_hitters = pd.DataFrame(cursor.fetchall(), columns=['player_id', 'name', 'proj_r', 'proj_hr', 'proj_tb', 'proj_sbn', 'proj_ab', 'proj_avg', 'proj_ops', 'eligible_positions', 'proj_pr'])
        # Select all free agent hitters (not on the current team)
        cursor = conn.execute(text("""
            SELECT player_id, name, proj_r, proj_hr, proj_tb, proj_sbn, proj_ab, proj_avg, proj_ops, eligible_positions, proj_pr
            FROM hitters
            WHERE team IS NULL OR team != :team_name
        """), {"team_name": team_name})
        free_agents = pd.DataFrame(cursor.fetchall(), columns=['player_id', 'name', 'proj_r', 'proj_hr', 'proj_tb', 'proj_sbn', 'proj_ab', 'proj_avg', 'proj_ops', 'eligible_positions', 'proj_pr'])
        selected_team_stats = all_teams_df[all_teams_df['team_name'] == team_name].iloc[0]
        original_points = team_totals[team_name]
        for idx, player in free_agents.iterrows():
            # Add the free agent to the team's hitters
            new_roster = pd.concat([team_hitters, pd.DataFrame([player])], ignore_index=True)
            # Use optimal roster selection
            optimal_roster = select_optimal_roster(new_roster)
            if optimal_roster.empty:
                continue
            # Calculate new team stats from optimal roster
            total_ab = optimal_roster['proj_ab'].sum()
            new_stats = {
                'r': optimal_roster['proj_r'].sum(),
                'hr': optimal_roster['proj_hr'].sum(),
                'tb': optimal_roster['proj_tb'].sum(),
                'sbn': optimal_roster['proj_sbn'].sum(),
                'avg': (optimal_roster['proj_avg'] * optimal_roster['proj_ab']).sum() / total_ab if total_ab > 0 else 0,
                'ops': (optimal_roster['proj_ops'] * optimal_roster['proj_ab']).sum() / total_ab if total_ab > 0 else 0,
            }
            # Update all_teams_df for this team
            new_teams_df = all_teams_df.copy()
            for col in ['r', 'hr', 'tb', 'sbn', 'avg', 'ops']:
                new_teams_df.loc[new_teams_df['team_name'] == team_name, col] = new_stats[col]
            new_points, new_category_points = calculate_points(new_teams_df, return_category_points=True)
            new_total = new_points[team_name]
            impact = new_total - original_points
            # Logging for key players
            if player['name'].strip().lower() in ['aaron judge', 'shohei ohtani']:
                logger.info(f"\n--- Impact Calculation for {player['name']} on team {team_name} ---")
                logger.info(f"Player stats: r={player['proj_r']}, hr={player['proj_hr']}, tb={player['proj_tb']}, sbn={player['proj_sbn']}, ab={player['proj_ab']}, avg={player['proj_avg']}, ops={player['proj_ops']}")
                logger.info(f"Original team stats: {selected_team_stats.to_dict()}")
                logger.info(f"Updated team stats after adding {player['name']}: {new_stats}")
                logger.info(f"Impact for {player['name']} on {team_name}: {impact}\n---\n")
            elif idx % 100 == 0:
                logger.info(f"Processed {idx} free agent impacts for team {team_name}")
            cursor = conn.execute(text("""
                INSERT INTO hitter_impacts (player_id, team_name, impact, stat_type)
                VALUES (:player_id, :team_name, :impact, 'projected')
            """), {"player_id": player['player_id'], "team_name": team_name, "impact": impact})
    conn.commit()

if __name__ == "__main__":
    main() 