from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import os
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_points(stats_df, return_category_points=False):
    """Calculate roto points for each category with proper tie handling."""
    categories = {
        'k': 1,      # high is best
        'qs': 1,     # high is best
        'svhd': 1,   # high is best
        'k_bb': 1,   # high is best
        'era': -1,   # low is best
        'whip': -1   # low is best
    }
    n_teams = len(stats_df)
    team_points = {team: 0 for team in stats_df['team_name']}
    category_points = {cat: {} for cat in categories}
    for category, direction in categories.items():
        if category in stats_df.columns:
            # Sort for ranking
            ascending = direction == -1
            ranked = stats_df[["team_name", category]].sort_values(category, ascending=ascending)
            # Assign ranks (1 for best, n for worst), handle ties by averaging
            ranked['rank'] = ranked[category].rank(method='min', ascending=ascending)
            # Points: 10 for best, 1 for worst
            ranked['points'] = n_teams - ranked['rank'] + 1
            for _, row in ranked.iterrows():
                team_points[row['team_name']] += row['points']
                category_points[category][row['team_name']] = row['points']
    if return_category_points:
        return team_points, category_points
    return team_points

def main():
    engine = create_engine(os.environ['DATABASE_URL'])
    # Load all teams' projected stats
    # We need to calculate projected stats by summing up all pitchers' projected stats for each team
    projected_stats_query = """
        WITH team_projected_stats AS (
            SELECT 
                team,
                SUM(proj_k) as k,
                SUM(proj_qs) as qs,
                SUM(proj_svhd) as svhd,
                SUM(proj_ip) as ip,
                SUM(proj_ip * proj_era) as weighted_era,
                SUM(proj_ip * proj_whip) as weighted_whip,
                SUM(proj_k) as total_k,
                SUM(proj_bb) as total_bb
            FROM pitchers 
            WHERE team != 'Free Agent'
            GROUP BY team
        )
        SELECT 
            team as team_name,
            k,
            qs,
            svhd,
            ip,
            CASE 
                WHEN ip > 0 THEN weighted_era / ip 
                ELSE 0 
            END as era,
            CASE 
                WHEN ip > 0 THEN weighted_whip / ip 
                ELSE 0 
            END as whip,
            CASE 
                WHEN total_bb > 0 THEN total_k / total_bb 
                ELSE 0 
            END as k_bb
        FROM team_projected_stats
    """
    logger.info("Executing query for team projected stats")
    all_teams_df = pd.read_sql(projected_stats_query, engine)
    
    # Get existing columns in team_stats table
    with engine.connect() as conn:
        cursor = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'team_stats'"))
        existing_columns = [row[0] for row in cursor.fetchall()]
    
    # Add pitching stats columns to team_stats table if they don't exist
    new_columns = {
        'k': 'REAL',
        'qs': 'REAL',
        'svhd': 'REAL',
        'ip': 'REAL',
        'era': 'REAL',
        'whip': 'REAL',
        'k_bb': 'REAL',
        'k_points': 'REAL',
        'qs_points': 'REAL',
        'svhd_points': 'REAL',
        'era_points': 'REAL',
        'whip_points': 'REAL',
        'k_bb_points': 'REAL'
    }
    
    for col, type_ in new_columns.items():
        if col not in existing_columns:
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE team_stats ADD COLUMN {col} {type_}"))
            except Exception as e:
                logger.warning(f"Could not add column {col}: {e}")
    
    # Calculate points for each team
    points, category_points = calculate_points(all_teams_df, return_category_points=True)
    
    # Add points to the DataFrame
    all_teams_df['k_points'] = all_teams_df['team_name'].map(category_points['k'])
    all_teams_df['qs_points'] = all_teams_df['team_name'].map(category_points['qs'])
    all_teams_df['svhd_points'] = all_teams_df['team_name'].map(category_points['svhd'])
    all_teams_df['era_points'] = all_teams_df['team_name'].map(category_points['era'])
    all_teams_df['whip_points'] = all_teams_df['team_name'].map(category_points['whip'])
    all_teams_df['k_bb_points'] = all_teams_df['team_name'].map(category_points['k_bb'])
    all_teams_df['total_points'] = all_teams_df['team_name'].map(points)
    
    # Update team_stats table with pitching stats
    for _, row in all_teams_df.iterrows():
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE team_stats 
                SET k = :k, qs = :qs, svhd = :svhd, ip = :ip, era = :era, whip = :whip, k_bb = :k_bb,
                    k_points = :k_points, qs_points = :qs_points, svhd_points = :svhd_points, era_points = :era_points, 
                    whip_points = :whip_points, k_bb_points = :k_bb_points
                WHERE team_name = :team_name
            """), {
                'k': row['k'], 'qs': row['qs'], 'svhd': row['svhd'], 'ip': row['ip'], 'era': row['era'], 'whip': row['whip'], 'k_bb': row['k_bb'],
                'k_points': row['k_points'], 'qs_points': row['qs_points'], 'svhd_points': row['svhd_points'], 'era_points': row['era_points'],
                'whip_points': row['whip_points'], 'k_bb_points': row['k_bb_points'], 'team_name': row['team_name']
            })
    
    logger.info("Updated team_stats table with pitching stats")
    
    # Get all team names
    all_team_names = all_teams_df['team_name'].unique()

    # Get all pitchers
    all_pitchers_query = "SELECT * FROM pitchers"
    logger.info("Executing query for all pitchers")
    all_pitchers = pd.read_sql(all_pitchers_query, engine)

    impact_data = []

    for team_name in all_team_names:
        team_idx = all_teams_df[all_teams_df['team_name'] == team_name].index[0]
        team_stats = all_teams_df.loc[team_idx].copy()
        original_points, original_cat_points = calculate_points(all_teams_df, return_category_points=True)
        original_total = original_points.get(team_name, 0)
        logger.info(f"Original total points for {team_name}: {original_total}")
        logger.info(f"{team_name} per-category points: " + ", ".join(f"{cat}: {original_cat_points[cat].get(team_name, 0):.2f}" for cat in original_cat_points))
        # Select all pitchers not on the current team
        pitchers_not_on_team = all_pitchers[all_pitchers['team'] != team_name]
        for _, pitcher in pitchers_not_on_team.iterrows():
            new_teams_df = all_teams_df.copy()
            new_stats = team_stats.copy()
            new_stats['k'] += pitcher['proj_k']
            new_stats['qs'] += pitcher['proj_qs']
            new_stats['svhd'] += pitcher['proj_svhd']
            new_ip = new_stats['ip'] + pitcher['proj_ip']
            new_stats['era'] = ((team_stats['era'] * team_stats['ip']) + pitcher['proj_era'] * pitcher['proj_ip']) / new_ip
            new_stats['whip'] = ((team_stats['whip'] * team_stats['ip']) + pitcher['proj_whip'] * pitcher['proj_ip']) / new_ip
            if pitcher['proj_k_bb'] is not None:
                new_stats['k_bb'] = ((team_stats['k_bb'] * team_stats['ip']) + pitcher['proj_k_bb'] * pitcher['proj_ip']) / new_ip
            else:
                new_stats['k_bb'] = (new_stats['k'] + pitcher['proj_k']) / (new_stats['bb'] + pitcher['proj_bb'])
            new_stats['ip'] = new_ip
            new_teams_df.loc[team_idx] = new_stats
            new_points, new_cat_points = calculate_points(new_teams_df, return_category_points=True)
            new_total = new_points.get(team_name, 0)
            point_diff = new_total - original_total
            cat_logs = []
            for cat in original_cat_points:
                orig = original_cat_points[cat].get(team_name, 0)
                new = new_cat_points[cat].get(team_name, 0)
                diff = new - orig
                cat_logs.append(f"{cat}: {orig:.2f} -> {new:.2f} (Δ {diff:+.2f})")
            logger.info(f"Pitcher {pitcher['name']}: Total Impact: {point_diff:+.2f} | " + ", ".join(cat_logs))
            impact_data.append({
                'player_id': pitcher['player_id'],
                'team_name': team_name,
                'impact': point_diff
            })

    if not impact_data:
        logger.error("No impacts were calculated")
        return

    with engine.connect() as conn:
        conn.execute(text('DROP TABLE IF EXISTS pitcher_impacts'))
        conn.execute(text('''
        CREATE TABLE pitcher_impacts (
            player_id INTEGER,
            team_name TEXT,
            impact REAL,
            stat_type TEXT DEFAULT 'projected',
            PRIMARY KEY (player_id, team_name, stat_type)
        )
        '''))

    impact_df = pd.DataFrame(impact_data)
    impact_df['stat_type'] = 'projected'
    impact_df.to_sql('pitcher_impacts', engine, if_exists='replace', index=False)

    logger.info(f"Successfully calculated and stored impacts for {len(impact_data)} pitcher-team pairs")

if __name__ == "__main__":
    main() 