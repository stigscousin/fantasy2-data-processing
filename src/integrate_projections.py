from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import sqlite3
import re
from typing import Dict, List
import logging
import unicodedata
import numpy as np
from scipy import stats
from sqlalchemy import create_engine
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Z_SCORE_CAP = 5.326  # 99.99999th percentile

def clean_name(name: str) -> str:
    """Clean player names to improve matching."""
    # Convert to lowercase first
    name = name.lower()
    
    # Remove accents and special characters, keeping basic letters
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove remaining special characters except spaces and periods
    name = re.sub(r'[^a-z\s\.]', '', name)
    
    # Remove common suffixes
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)$', '', name)
    
    # Remove middle names/initials
    name = re.sub(r'\s+[a-z]\.?\s+', ' ', name)
    
    # Remove extra spaces and trim
    name = ' '.join(name.split())
    
    # Handle "Last, First" format
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"
    
    return name

def clean_team(team: str) -> str:
    """Clean team names to improve matching."""
    team_map = {
        'NYY': 'NY Yankees',
        'BOS': 'Boston',
        'TBR': 'Tampa Bay',
        'TOR': 'Toronto',
        'BAL': 'Baltimore',
        'CHW': 'Chi White Sox',
        'CLE': 'Cleveland',
        'DET': 'Detroit',
        'KCR': 'Kansas City',
        'MIN': 'Minnesota',
        'HOU': 'Houston',
        'LAA': 'LA Angels',
        'OAK': 'Oakland',
        'SEA': 'Seattle',
        'TEX': 'Texas',
        'ATL': 'Atlanta',
        'MIA': 'Miami',
        'NYM': 'NY Mets',
        'PHI': 'Philadelphia',
        'WSN': 'Washington',
        'CHC': 'Chi Cubs',
        'CIN': 'Cincinnati',
        'MIL': 'Milwaukee',
        'PIT': 'Pittsburgh',
        'STL': 'St. Louis',
        'ARI': 'Arizona',
        'COL': 'Colorado',
        'LAD': 'LA Dodgers',
        'SDP': 'San Diego',
        'SFG': 'San Francisco'
    }
    return team_map.get(team, team)

def calculate_projection_pr_values(stats_df: pd.DataFrame, is_hitter: bool) -> pd.DataFrame:
    """Calculate PR (Player Rating) values for each player's projections based on z-scores of their stats."""
    df = stats_df.copy()
    
    if is_hitter:
        # Calculate weighted averages for rate stats
        weights = df['ab'].fillna(0)
        if weights.sum() > 0:
            league_avg = np.average(df['avg'].fillna(0), weights=weights)
            league_ops = np.average(df['ops'].fillna(0), weights=weights)
        else:
            league_avg = 0
            league_ops = 0
        
        # Calculate weighted differences
        df['avg_diff'] = (df['avg'].fillna(0) - league_avg) * weights
        df['ops_diff'] = (df['ops'].fillna(0) - league_ops) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['r', 'hr', 'tb', 'sbn']:
            df[f'{stat}_z'] = stats.zscore(df[stat].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        df['avg_z'] = stats.zscore(df['avg_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['ops_z'] = stats.zscore(df['ops_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate total PR
        pr_columns = ['r_z', 'hr_z', 'tb_z', 'sbn_z', 'avg_z', 'ops_z']
        df['pr'] = df[pr_columns].sum(axis=1)
        
    else:  # Pitchers
        # Calculate weighted averages for rate stats
        weights = df['ip'].fillna(0)
        if weights.sum() > 0:
            league_era = np.average(df['era'].fillna(0), weights=weights)
            league_whip = np.average(df['whip'].fillna(0), weights=weights)
            
            # Calculate league totals and average K/BB ratio
            total_league_k = np.sum(df['k'].fillna(0))
            total_league_bb = np.sum(df['bb'].fillna(0))
            league_kbb = total_league_k / total_league_bb if total_league_bb > 0 else 0
            
            # Calculate each pitcher's K/BB ratio
            df['k_bb'] = df['k'].fillna(0) / df['bb'].replace(0, np.nan)
            df['k_bb'] = df['k_bb'].fillna(0)
            # Calculate weighted difference for K/BB
            df['k_bb_diff'] = (df['k_bb'] - league_kbb) * weights
        else:
            league_era = 0
            league_whip = 0
            league_kbb = 0
            df['k_bb'] = 0
            df['k_bb_diff'] = 0
        
        # Calculate weighted differences (inverse for ERA and WHIP since lower is better)
        df['era_diff'] = (league_era - df['era'].fillna(0)) * weights
        df['whip_diff'] = (league_whip - df['whip'].fillna(0)) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['k', 'qs', 'svhd']:
            df[f'{stat}_z'] = stats.zscore(df[stat].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        df['era_z'] = stats.zscore(df['era_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['whip_z'] = stats.zscore(df['whip_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['k_bb_z'] = stats.zscore(df['k_bb_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate total PR
        pr_columns = ['k_z', 'qs_z', 'era_z', 'whip_z', 'k_bb_z', 'svhd_z']
        df['pr'] = df[pr_columns].sum(axis=1)
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def calculate_current_pr_values(stats_df: pd.DataFrame, is_hitter: bool) -> pd.DataFrame:
    """Calculate PR (Player Rating) values for each player's current stats based on z-scores."""
    df = stats_df.copy()
    
    if is_hitter:
        # Calculate weighted averages for rate stats
        weights = df['ab'].fillna(0)
        if weights.sum() > 0:
            league_avg = np.average(df['avg'].fillna(0), weights=weights)
            league_ops = np.average(df['ops'].fillna(0), weights=weights)
        else:
            league_avg = 0
            league_ops = 0
        
        # Calculate weighted differences
        df['avg_diff'] = (df['avg'].fillna(0) - league_avg) * weights
        df['ops_diff'] = (df['ops'].fillna(0) - league_ops) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['r', 'hr', 'tb', 'sbn']:
            df[f'{stat}_z'] = stats.zscore(df[stat].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        df['avg_z'] = stats.zscore(df['avg_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['ops_z'] = stats.zscore(df['ops_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate total PR
        pr_columns = ['r_z', 'hr_z', 'tb_z', 'sbn_z', 'avg_z', 'ops_z']
        df['current_pr'] = df[pr_columns].sum(axis=1)
        
    else:  # Pitchers
        # Calculate weighted averages for rate stats
        weights = df['ip'].fillna(0)
        if weights.sum() > 0:
            league_era = np.average(df['era'].fillna(0), weights=weights)
            league_whip = np.average(df['whip'].fillna(0), weights=weights)
            
            # Calculate league totals and average K/BB ratio
            total_league_k = np.sum(df['k'].fillna(0))
            total_league_bb = np.sum(df['bb'].fillna(0))
            league_kbb = total_league_k / total_league_bb if total_league_bb > 0 else 0
            
            # Calculate each pitcher's K/BB ratio
            df['k_bb'] = df['k'].fillna(0) / df['bb'].replace(0, np.nan)
            df['k_bb'] = df['k_bb'].fillna(0)
            # Calculate weighted difference for K/BB
            df['k_bb_diff'] = (df['k_bb'] - league_kbb) * weights
        else:
            league_era = 0
            league_whip = 0
            league_kbb = 0
            df['k_bb'] = 0
            df['k_bb_diff'] = 0
        
        # Calculate weighted differences (inverse for ERA and WHIP since lower is better)
        df['era_diff'] = (league_era - df['era'].fillna(0)) * weights
        df['whip_diff'] = (league_whip - df['whip'].fillna(0)) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['k', 'qs', 'svhd']:
            df[f'{stat}_z'] = stats.zscore(df[stat].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        df['era_z'] = stats.zscore(df['era_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['whip_z'] = stats.zscore(df['whip_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        df['k_bb_z'] = stats.zscore(df['k_bb_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate total PR
        pr_columns = ['k_z', 'qs_z', 'era_z', 'whip_z', 'k_bb_z', 'svhd_z']
        df['current_pr'] = df[pr_columns].sum(axis=1)
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def integrate_projections():
    # Connect to the database
    db_url = os.environ['DATABASE_URL']
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    engine = create_engine(db_url)
    
    # Read the projection files
    logger.info("Reading projection files...")
    
    # Define dtypes for projection files
    hitter_dtypes = {
        'Name': 'str',
        'Team': 'str',
        'G': 'float32',
        'PA': 'float32',
        'AB': 'float32',
        'H': 'float32',
        '2B': 'float32',
        '3B': 'float32',
        'HR': 'float32',
        'R': 'float32',
        'RBI': 'float32',
        'BB': 'float32',
        'SO': 'float32',
        'SB': 'float32',
        'CS': 'float32',
        'AVG': 'float32',
        'OPS': 'float32',
        'wOBA': 'float32',
        'wRC+': 'float32',
        'WAR': 'float32',
        'MLBAMID': 'str'
    }

    pitcher_dtypes = {
        'Name': 'str',
        'Team': 'str',
        'G': 'float32',
        'GS': 'float32',
        'IP': 'float32',
        'SO': 'float32',
        'BB': 'float32',
        'HR': 'float32',
        'ERA': 'float32',
        'WHIP': 'float32',
        'W': 'float32',
        'L': 'float32',
        'SV': 'float32',
        'HLD': 'float32',
        'QS': 'float32',
        'K/BB': 'float32',
        'MLBAMID': 'str'
    }
    
    # Read and validate hitter projections
    logger.info("Reading hitter projections...")
    hitter_proj = pd.read_csv('projections/fangraphs-leaderboard-projections-batters.csv', 
                             dtype=hitter_dtypes,
                             usecols=list(hitter_dtypes.keys()))
    hitter_proj.columns = hitter_proj.columns.str.lower()
    
    # Read and validate pitcher projections
    logger.info("Reading pitcher projections...")
    pitcher_proj = pd.read_csv('projections/fangraphs-leaderboard-projections-pitchers.csv',
                              dtype=pitcher_dtypes,
                              usecols=list(pitcher_dtypes.keys()))
    pitcher_proj.columns = pitcher_proj.columns.str.lower()
    
    # Clean team names in projections
    hitter_proj['team'] = hitter_proj['team'].apply(clean_team)
    pitcher_proj['team'] = pitcher_proj['team'].apply(clean_team)
    
    # Log initial data validation
    logger.info(f"Number of hitters in projections: {len(hitter_proj)}")
    logger.info(f"Number of pitchers in projections: {len(pitcher_proj)}")
    
    # Process hitters
    logger.info("Processing hitter projections...")
    # Calculate Total Bases (TB) from available stats
    hitter_proj['tb'] = (
        hitter_proj['h'].fillna(0) + 
        hitter_proj['2b'].fillna(0) + 
        (hitter_proj['3b'].fillna(0) * 2) + 
        (hitter_proj['hr'].fillna(0) * 3)
    )
    
    # Calculate Stolen Bases Net (SBN)
    hitter_proj['sbn'] = hitter_proj['sb'].fillna(0) - hitter_proj['cs'].fillna(0)
    
    # Read existing stats
    logger.info("Reading existing stats...")
    existing_hitters = pd.read_csv('data/hitter_stats.csv', 
                                  dtype={'mlbamid': 'str'},
                                  low_memory=False)
    existing_hitters.columns = existing_hitters.columns.str.lower()
    
    # Calculate current PR values
    logger.info("Calculating current PR values...")
    existing_hitters = calculate_current_pr_values(existing_hitters, is_hitter=True)
    
    # Calculate projection PR values
    logger.info("Calculating projection PR values...")
    hitter_proj = calculate_projection_pr_values(hitter_proj, is_hitter=True)
    
    # Clean team names in existing stats
    existing_hitters['team'] = existing_hitters['team'].apply(clean_team)
    
    # Log initial state of existing data
    logger.info(f"Number of existing players: {len(existing_hitters)}")
    logger.info(f"Sample of existing data:\n{existing_hitters.head().to_string()}")
    
    # Clean up any existing projection columns to prevent duplicates
    proj_columns = [col for col in existing_hitters.columns if col.startswith('proj_')]
    existing_hitters = existing_hitters.drop(columns=proj_columns, errors='ignore')
    
    # Rename columns in hitter_proj to match existing_hitters
    hitter_proj = hitter_proj.rename(columns={
        'Name': 'name',
        'Team': 'team',
        'MLBAMID': 'mlbamid',
        'wRC+': 'wRC_plus'
    })
    
    # Add 'proj_' prefix to all stat columns
    stat_columns = [col for col in hitter_proj.columns if col not in ['name', 'team', 'mlbamid']]
    for col in stat_columns:
        hitter_proj = hitter_proj.rename(columns={col: f'proj_{col}'})
    
    # Merge projections with existing stats by mlbamid (left join, so all existing_hitters are kept)
    merged_stats = pd.merge(existing_hitters, hitter_proj, 
                           on=['mlbamid'], 
                           how='left',
                           suffixes=('', '_proj'))

    # Identify hitters still unmatched (no projection after mlbamid merge)
    unmatched_mask = merged_stats['proj_r'].isna()
    unmatched_hitters = merged_stats[unmatched_mask].copy()
    matched_hitters = merged_stats[~unmatched_mask].copy()

    logger.info(f"Hitters matched on mlbamid: {len(matched_hitters)}")
    logger.info(f"Hitters unmatched on mlbamid: {len(unmatched_hitters)}")
    if len(unmatched_hitters) > 0:
        logger.info(f"Sample of unmatched hitters (by mlbamid):\n{unmatched_hitters[['name', 'team', 'mlbamid']].head().to_string()}")

    # Name-based matching for only those still unmatched
    if len(unmatched_hitters) > 0:
        logger.info(f"Found {len(unmatched_hitters)} hitters without mlbamid matches. Attempting name-based matching...")
        # Clean names for matching
        unmatched_hitters['clean_name'] = unmatched_hitters['name'].apply(clean_name)
        hitter_proj['clean_name'] = hitter_proj['name'].apply(clean_name)
        logger.info(f"Sample of unmatched hitter clean names: {unmatched_hitters['clean_name'].head().tolist()}")
        logger.info(f"Sample of projection clean names: {hitter_proj['clean_name'].head().tolist()}")
        # Merge by clean_name (left join, so all unmatched_hitters are kept)
        name_matched = pd.merge(
            unmatched_hitters.drop([col for col in unmatched_hitters.columns if col.startswith('proj_')], axis=1, errors='ignore'),
            hitter_proj,
            on='clean_name',
            how='left',
            suffixes=('', '_proj')
        )
        # Only fill in columns that already start with proj_ (avoid proj_proj_)
        proj_cols = [col for col in hitter_proj.columns if col.startswith('proj_')]
        for idx, row in name_matched.iterrows():
            orig_idx = merged_stats[(merged_stats['name'] == row['name']) & (merged_stats['proj_r'].isna())].index
            if not orig_idx.empty:
                for col in proj_cols:
                    if col in name_matched.columns:
                        merged_stats.loc[orig_idx, col] = row[col]
        # Log any still-unmatched hitters
        still_unmatched = merged_stats[merged_stats['proj_r'].isna()]
        logger.info(f"Hitters still unmatched after name matching: {len(still_unmatched)}")
        if len(still_unmatched) > 0:
            logger.info(f"Sample of still-unmatched hitters:\n{still_unmatched[['name', 'team', 'mlbamid']].head().to_string()}")
    
    # Calculate blending weights based on PA
    pa_weight = merged_stats['pa'].fillna(0) / 600  # Full season is roughly 600 PA
    pa_weight = pa_weight.clip(0, 1)  # Ensure weight is between 0 and 1
    proj_weight = 1 - pa_weight
    
    # List of stats to blend (excluding PR for special handling)
    stats_to_blend = ['avg', 'ops', 'r', 'hr', 'tb', 'sbn']
    
    # For blended stats, use projected AB
    merged_stats['blended_ab'] = merged_stats['proj_ab']
    merged_stats['blended_g'] = merged_stats['proj_g']
    
    # Perform blending for each stat
    for stat in stats_to_blend:
        real_stat = merged_stats[stat].fillna(0)
        proj_stat = merged_stats[f'proj_{stat}'].fillna(0)
        merged_stats[f'blended_{stat}'] = (real_stat * pa_weight) + (proj_stat * proj_weight)

    # Set is_hitter to True since we're processing hitter data
    is_hitter = True

    # Calculate z-scores for blended stats
    if is_hitter:
        # Calculate weighted averages for rate stats
        weights = merged_stats['blended_ab'].fillna(0)  # Use blended AB for weights
        if weights.sum() > 0:
            league_avg = np.average(merged_stats['blended_avg'].fillna(0), weights=weights)
            league_ops = np.average(merged_stats['blended_ops'].fillna(0), weights=weights)
        else:
            league_avg = 0
            league_ops = 0
        
        # Calculate weighted differences
        merged_stats['blended_avg_diff'] = (merged_stats['blended_avg'].fillna(0) - league_avg) * weights
        merged_stats['blended_ops_diff'] = (merged_stats['blended_ops'].fillna(0) - league_ops) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['r', 'hr', 'tb', 'sbn']:
            merged_stats[f'blended_{stat}_z'] = stats.zscore(merged_stats[f'blended_{stat}'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        merged_stats['blended_avg_z'] = stats.zscore(merged_stats['blended_avg_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        merged_stats['blended_ops_z'] = stats.zscore(merged_stats['blended_ops_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
    else:  # Pitchers
        # Calculate weighted averages for rate stats
        weights = merged_stats['ip'].fillna(0)
        if weights.sum() > 0:
            league_era = np.average(merged_stats['blended_era'].fillna(0), weights=weights)
            league_whip = np.average(merged_stats['blended_whip'].fillna(0), weights=weights)
        else:
            league_era = 0
            league_whip = 0
        
        # Calculate weighted differences (inverse for ERA and WHIP since lower is better)
        merged_stats['blended_era_diff'] = (league_era - merged_stats['blended_era'].fillna(0)) * weights
        merged_stats['blended_whip_diff'] = (league_whip - merged_stats['blended_whip'].fillna(0)) * weights
        
        # Calculate z-scores for counting stats
        for stat in ['k', 'qs', 'svhd']:
            merged_stats[f'blended_{stat}_z'] = stats.zscore(merged_stats[f'blended_{stat}'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        
        # Calculate z-scores for weighted rate stats
        merged_stats['blended_era_z'] = stats.zscore(merged_stats['blended_era_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
        merged_stats['blended_whip_z'] = stats.zscore(merged_stats['blended_whip_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)

    # Special case for PR: blend current_pr with proj_pr
    real_pr = merged_stats['current_pr'].fillna(0)
    proj_pr = merged_stats['proj_pr'].fillna(0)
    merged_stats['blended_pr'] = (real_pr * pa_weight) + (proj_pr * proj_weight)
    
    # Clean up column names
    # Remove any columns with _x or _y suffixes
    merged_stats = merged_stats.loc[:, ~merged_stats.columns.str.endswith(('_x', '_y'))]
    
    # Remove any duplicate name columns
    name_cols = [col for col in merged_stats.columns if col.startswith('name_')]
    if len(name_cols) > 1:
        merged_stats = merged_stats.drop(columns=name_cols[1:])
    
    # Remove any duplicate team columns
    team_cols = [col for col in merged_stats.columns if col.startswith('team_')]
    if len(team_cols) > 1:
        merged_stats = merged_stats.drop(columns=team_cols[1:])
    
    # Drop any remaining columns with _proj suffix that aren't in the database schema
    proj_suffix_cols = [col for col in merged_stats.columns if col.endswith('_proj')]
    merged_stats = merged_stats.drop(columns=proj_suffix_cols)
    
    # Save the updated stats
    logger.info(f"merged_stats columns: {list(merged_stats.columns)}")
    logger.info("Saving updated stats...")
    merged_stats.to_csv('data/hitter_stats.csv', index=False)
    
    # Update the database with blended stats
    logger.info("Updating database with blended stats...")
    merged_stats.to_sql('hitters', engine, if_exists='replace', index=False)
    
    # Process pitchers
    logger.info("Processing pitcher projections...")
    # Calculate SVHD (Saves + Holds)
    pitcher_proj['svhd'] = pitcher_proj['sv'].fillna(0) + pitcher_proj['hld'].fillna(0)
    
    # Rename SO to K for consistency
    pitcher_proj = pitcher_proj.rename(columns={'so': 'k'})
    # Rename K/BB to K_BB for SQL compatibility
    pitcher_proj = pitcher_proj.rename(columns={'k/bb': 'k_bb'})
    # Now fill missing K_BB with 0 (after renaming)
    pitcher_proj['k_bb'] = pitcher_proj['k_bb'].fillna(0)
    # Use existing QS from projections
    pitcher_proj['qs'] = pitcher_proj['qs'].fillna(0)
    
    # Read existing pitcher stats
    existing_pitchers = pd.read_csv('data/pitcher_stats.csv',
                                  dtype={'mlbamid': 'str'},
                                  low_memory=False)
    existing_pitchers.columns = existing_pitchers.columns.str.lower()
    
    # Calculate current PR values for pitchers
    logger.info("Calculating current PR values for pitchers...")
    existing_pitchers = calculate_current_pr_values(existing_pitchers, is_hitter=False)
    
    # Calculate projection PR values for pitchers
    logger.info("Calculating projection PR values for pitchers...")
    pitcher_proj = calculate_projection_pr_values(pitcher_proj, is_hitter=False)
    
    # Clean team names in existing stats
    existing_pitchers['team'] = existing_pitchers['team'].apply(clean_team)
    
    # Clean up any existing projection columns to prevent duplicates
    proj_columns = [col for col in existing_pitchers.columns if col.startswith('proj_')]
    existing_pitchers = existing_pitchers.drop(columns=proj_columns, errors='ignore')
    
    # Rename columns in pitcher_proj to match existing_pitchers
    pitcher_proj = pitcher_proj.rename(columns={
        'Name': 'name',
        'Team': 'team',
        'MLBAMID': 'mlbamid'
    })
    
    # Add 'proj_' prefix to all stat columns
    stat_columns = [col for col in pitcher_proj.columns if col not in ['name', 'team', 'mlbamid']]
    for col in stat_columns:
        pitcher_proj = pitcher_proj.rename(columns={col: f'proj_{col}'})
    
    # Calculate K/BB z-scores for projections
    weights = pitcher_proj['proj_ip'].fillna(0)
    if weights.sum() > 0:
        # Calculate league totals and average K/BB ratio
        total_league_k = np.sum(pitcher_proj['proj_k'].fillna(0))
        total_league_bb = np.sum(pitcher_proj['proj_bb'].fillna(0))
        league_kbb = total_league_k / total_league_bb if total_league_bb > 0 else 0
        
        # Calculate each pitcher's K/BB ratio
        pitcher_proj['proj_k_bb'] = pitcher_proj['proj_k'].fillna(0) / pitcher_proj['proj_bb'].replace(0, np.nan)
        pitcher_proj['proj_k_bb'] = pitcher_proj['proj_k_bb'].fillna(0)
        # Calculate weighted difference for K/BB
        pitcher_proj['proj_k_bb_diff'] = (pitcher_proj['proj_k_bb'] - league_kbb) * weights
        # Calculate z-scores for K/BB diff
        pitcher_proj['proj_k_bb_z'] = stats.zscore(pitcher_proj['proj_k_bb_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
    else:
        pitcher_proj['proj_k_bb'] = 0
        pitcher_proj['proj_k_bb_diff'] = 0
        pitcher_proj['proj_k_bb_z'] = 0
    
    # Log columns after prefixing
    logger.info(f"Columns in pitcher_proj after prefixing: {pitcher_proj.columns.tolist()}")
    
    # Merge projections with existing stats
    merged_pitcher_stats = pd.merge(existing_pitchers, pitcher_proj,
                                  on=['mlbamid'],
                                  how='left',
                                  suffixes=('', '_proj'))
    
    # Find pitchers that didn't match on mlbamid
    unmatched_pitchers = merged_pitcher_stats[merged_pitcher_stats['proj_k'].isna()].copy()
    matched_pitchers = merged_pitcher_stats[~merged_pitcher_stats['proj_k'].isna()].copy()
    
    logger.info(f"Pitchers matched on mlbamid: {len(matched_pitchers)}")
    logger.info(f"Pitchers unmatched on mlbamid: {len(unmatched_pitchers)}")
    if len(unmatched_pitchers) > 0:
        logger.info(f"Sample of unmatched pitchers (by mlbamid):\n{unmatched_pitchers[['name', 'team', 'mlbamid']].head().to_string()}")
    
    if len(unmatched_pitchers) > 0:
        logger.info(f"Found {len(unmatched_pitchers)} pitchers without mlbamid matches. Attempting name-based matching...")
        
        # Clean names for matching
        unmatched_pitchers['clean_name'] = unmatched_pitchers['name'].apply(clean_name)
        pitcher_proj['clean_name'] = pitcher_proj['name'].apply(clean_name)
        logger.info(f"Sample of unmatched pitcher clean names: {unmatched_pitchers['clean_name'].head().tolist()}")
        logger.info(f"Sample of projection clean names: {pitcher_proj['clean_name'].head().tolist()}")
        
        # Try to match on cleaned names
        name_matched = pd.merge(
            unmatched_pitchers,
            pitcher_proj,
            on='clean_name',
            how='left',
            suffixes=('', '_proj')
        )
        
        # Get only the rows with valid projections
        name_matched = name_matched[~name_matched['proj_k'].isna()]
        logger.info(f"Pitchers matched by name: {len(name_matched)}")
        if len(name_matched) > 0:
            logger.info(f"Sample of name-matched pitchers:\n{name_matched[['name', 'team', 'clean_name', 'proj_k']].head().to_string()}")
        else:
            logger.warning("No pitchers matched by name.")
        
        # Clean up duplicate columns before concatenation
        name_matched = name_matched.loc[:, ~name_matched.columns.str.endswith(('_x', '_y'))]
        
        # Remove any duplicate name columns
        name_cols = [col for col in name_matched.columns if col.startswith('name_')]
        if len(name_cols) > 1:
            name_matched = name_matched.drop(columns=name_cols[1:])
        
        # Remove any duplicate team columns
        team_cols = [col for col in name_matched.columns if col.startswith('team_')]
        if len(team_cols) > 1:
            name_matched = name_matched.drop(columns=team_cols[1:])
        
        # Combine matched pitchers with name-matched pitchers
        merged_pitcher_stats = pd.concat([matched_pitchers, name_matched], ignore_index=True)
        
        # Log any still-unmatched pitchers
        still_unmatched = merged_pitcher_stats[merged_pitcher_stats['proj_k'].isna()]
        logger.info(f"Pitchers still unmatched after name matching: {len(still_unmatched)}")
        if len(still_unmatched) > 0:
            logger.info(f"Sample of still-unmatched pitchers:\n{still_unmatched[['name', 'team', 'mlbamid']].head().to_string()}")
    
    # Calculate blending weights based on IP
    ip_weight = merged_pitcher_stats['ip'].fillna(0) / 200  # Full season is roughly 200 IP
    ip_weight = ip_weight.clip(0, 1)  # Ensure weight is between 0 and 1
    proj_weight = 1 - ip_weight
    
    # List of stats to blend for pitchers
    pitcher_stats_to_blend = ['k', 'qs', 'era', 'whip', 'svhd']  # Remove K_BB from this list
    
    # Before blending loop, check DataFrame shape
    logger.info(f"Pitcher stats DataFrame shape before blending: {merged_pitcher_stats.shape}")
    if len(merged_pitcher_stats) == 0:
        logger.warning("No pitcher stats to blend! Skipping pitcher blending loop.")
    else:
        # Perform blending for each stat (except K_BB)
        for stat in pitcher_stats_to_blend:
            real_col = stat
            proj_col = f'proj_{stat}'
            # Ensure both columns exist as Series of NaNs if missing or if not a Series
            if real_col not in merged_pitcher_stats.columns or not isinstance(merged_pitcher_stats[real_col], pd.Series):
                logger.warning(f"Column {real_col} missing or not a Series. Forcibly setting to NaN Series.")
                merged_pitcher_stats[real_col] = pd.Series([np.nan] * len(merged_pitcher_stats), index=merged_pitcher_stats.index)
            if proj_col not in merged_pitcher_stats.columns or not isinstance(merged_pitcher_stats[proj_col], pd.Series):
                logger.warning(f"Column {proj_col} missing or not a Series. Forcibly setting to NaN Series.")
                merged_pitcher_stats[proj_col] = pd.Series([np.nan] * len(merged_pitcher_stats), index=merged_pitcher_stats.index)
            # Now convert to numeric
            merged_pitcher_stats[real_col] = pd.to_numeric(merged_pitcher_stats[real_col], errors='coerce').fillna(0)
            merged_pitcher_stats[proj_col] = pd.to_numeric(merged_pitcher_stats[proj_col], errors='coerce').fillna(0)
            # Calculate blended value
            merged_pitcher_stats[f'blended_{stat}'] = (
                merged_pitcher_stats[real_col] * ip_weight + 
                merged_pitcher_stats[proj_col] * proj_weight
            )
    # Directly blend K/BB ratios
    merged_pitcher_stats['blended_k_bb'] = (
        merged_pitcher_stats['k_bb'] * ip_weight + merged_pitcher_stats['proj_k_bb'] * proj_weight
    )
    
    # Calculate z-scores for blended stats
    # Calculate weighted averages for rate stats
    weights = merged_pitcher_stats['ip'].fillna(0)
    if weights.sum() > 0:
        league_era = np.average(merged_pitcher_stats['blended_era'].fillna(0), weights=weights)
        league_whip = np.average(merged_pitcher_stats['blended_whip'].fillna(0), weights=weights)
        
        # Calculate league totals and average K/BB ratio
        total_league_k = np.sum(merged_pitcher_stats['blended_k'].fillna(0))
        total_league_bb = np.sum(merged_pitcher_stats['bb'].fillna(0))
        league_kbb = total_league_k / total_league_bb if total_league_bb > 0 else 0
        
        # Calculate weighted difference for blended K/BB
        merged_pitcher_stats['blended_k_bb_diff'] = (merged_pitcher_stats['blended_k_bb'] - league_kbb) * weights
    else:
        league_era = 0
        league_whip = 0
        league_kbb = 0
        merged_pitcher_stats['blended_k_bb_diff'] = 0

    # Calculate weighted differences (inverse for ERA and WHIP since lower is better)
    merged_pitcher_stats['blended_era_diff'] = (league_era - merged_pitcher_stats['blended_era'].fillna(0)) * weights
    merged_pitcher_stats['blended_whip_diff'] = (league_whip - merged_pitcher_stats['blended_whip'].fillna(0)) * weights

    # Calculate z-scores for counting stats, only if the blended column exists
    for stat in ['k', 'qs', 'svhd']:
        col = f'blended_{stat}'
        if col in merged_pitcher_stats.columns:
            merged_pitcher_stats[f'{col}_z'] = stats.zscore(merged_pitcher_stats[col].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)

    # Calculate z-scores for weighted rate stats
    merged_pitcher_stats['blended_era_z'] = stats.zscore(merged_pitcher_stats['blended_era_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
    merged_pitcher_stats['blended_whip_z'] = stats.zscore(merged_pitcher_stats['blended_whip_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)
    merged_pitcher_stats['blended_k_bb_z'] = stats.zscore(merged_pitcher_stats['blended_k_bb_diff'].fillna(0)).clip(-Z_SCORE_CAP, Z_SCORE_CAP)

    # Calculate blended PR, only using available z-score columns
    pr_columns = [col for col in ['blended_k_z', 'blended_qs_z', 'blended_era_z', 'blended_whip_z', 'blended_k_bb_z', 'blended_svhd_z'] if col in merged_pitcher_stats.columns]
    merged_pitcher_stats['blended_pr'] = merged_pitcher_stats[pr_columns].sum(axis=1)
    
    # LOGGING FOR AARON CIVALE BLENDED STATS
    civale_row = merged_pitcher_stats[merged_pitcher_stats['name'].str.lower() == 'aaron civale'.lower()]
    if not civale_row.empty:
        logger.info('Blended stat calculations for Aaron Civale:')
        logger.info(civale_row[[col for col in merged_pitcher_stats.columns if col.startswith('blended_') or col in ['name','team','ip','proj_ip','k','proj_k','qs','proj_qs','era','proj_era','whip','proj_whip','k_bb','proj_k_bb','svhd','proj_svhd','current_pr','proj_pr']]].to_string(index=False))
        logger.info(f"ip_weight: {civale_row['ip'].values[0]/200 if not np.isnan(civale_row['ip'].values[0]) else 'NaN'}  proj_weight: {1 - (civale_row['ip'].values[0]/200) if not np.isnan(civale_row['ip'].values[0]) else 'NaN'}")
    else:
        logger.warning('Aaron Civale not found in merged_pitcher_stats for blended stat logging.')
    
    # Clean up column names for pitchers
    merged_pitcher_stats = merged_pitcher_stats.loc[:, ~merged_pitcher_stats.columns.str.endswith(('_x', '_y'))]
    
    # Remove any duplicate name columns
    name_cols = [col for col in merged_pitcher_stats.columns if col.startswith('name_')]
    if len(name_cols) > 1:
        merged_pitcher_stats = merged_pitcher_stats.drop(columns=name_cols[1:])
    
    # Remove any duplicate team columns
    team_cols = [col for col in merged_pitcher_stats.columns if col.startswith('team_')]
    if len(team_cols) > 1:
        merged_pitcher_stats = merged_pitcher_stats.drop(columns=team_cols[1:])
    
    # Drop the clean_name column as it's not in the database schema
    if 'clean_name' in merged_pitcher_stats.columns:
        merged_pitcher_stats = merged_pitcher_stats.drop(columns=['clean_name'])
    
    # Drop any remaining columns with _proj suffix that aren't in the database schema
    proj_suffix_cols = [col for col in merged_pitcher_stats.columns if col.endswith('_proj')]
    merged_pitcher_stats = merged_pitcher_stats.drop(columns=proj_suffix_cols)
    
    # Save the updated pitcher stats
    logger.info("Saving updated pitcher stats...")
    merged_pitcher_stats.to_csv('data/pitcher_stats.csv', index=False)
    
    # Update the database with blended pitcher stats
    logger.info("Updating database with blended pitcher stats...")
    merged_pitcher_stats.to_sql('pitchers', engine, if_exists='append', index=False)
    
    logger.info("Done!")

if __name__ == "__main__":
    integrate_projections() 