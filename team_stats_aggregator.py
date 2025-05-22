from espn_api.baseball import League
from espn_api.baseball.box_score import H2HCategoryBoxScore
from typing import Dict, List
import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

def aggregate_team_stats(league: League, start_week: int = 1, end_week: int = None) -> pd.DataFrame:
    """
    Aggregate team statistics from active matchup stats for a given range of weeks.
    
    Args:
        league: The league object
        start_week: First week to include (inclusive)
        end_week: Last week to include (inclusive). Defaults to current matchup period.
        
    Returns:
        DataFrame with team statistics
    """
    # If end_week is not specified, use the current matchup period
    if end_week is None:
        end_week = league.currentMatchupPeriod
        print(f"Using current matchup period {end_week} as end week")
    
    # Initialize team stats dictionary
    team_stats = {}
    
    # Initialize stats for each team
    for team in league.teams:
        team_id = team.team_id
        team_stats[team_id] = {
            'team_name': team.team_name,
            'R': 0,
            'HR': 0,
            'TB': 0,
            'SB': 0,
            'CS': 0,
            'H': 0,
            'AB': 0,
            'BB': 0,
            'HBP': 0,
            'SF': 0,
            'K': 0,
            'QS': 0,
            'ER': 0,
            'OUTS': 0,
            'P_BB': 0,
            'P_H': 0,
            'SVHD': 0,
            'OPS': 0,
            'OPS_AB': 0,  # Track ABs for OPS calculation
            'IP': 0,      # Innings Pitched
            'GS': 0       # Games Started
        }
    
    # Process each week's box scores
    for week in range(start_week, end_week + 1):
        # --- NEW: Per-week stats dict ---
        week_team_stats = {}
        for team in league.teams:
            team_id = team.team_id
            week_team_stats[team_id] = {
                'team_name': team.team_name,
                'R': 0,
                'HR': 0,
                'TB': 0,
                'SB': 0,
                'CS': 0,
                'H': 0,
                'AB': 0,
                'BB': 0,
                'HBP': 0,
                'SF': 0,
                'K': 0,
                'QS': 0,
                'ER': 0,
                'OUTS': 0,
                'P_BB': 0,
                'P_H': 0,
                'SVHD': 0,
                'OPS': 0,
                'OPS_AB': 0,
                'IP': 0,
                'GS': 0
            }
        box_scores = league.box_scores(matchup_period=week)
        for box_score in box_scores:
            if isinstance(box_score, H2HCategoryBoxScore):
                processed_teams = set()
                # Home team
                if box_score.home_team and box_score.home_team.team_id not in processed_teams:
                    team_id = box_score.home_team.team_id
                    stats = box_score.home_stats
                    if hasattr(box_score, 'home_lineup') and box_score.home_lineup:
                        for player in box_score.home_lineup:
                            if hasattr(player, 'slot_position') and player.slot_position == 'SP':
                                week_team_stats[team_id]['GS'] += 1
                    if 'R' in stats:
                        week_team_stats[team_id]['R'] += float(stats['R']['value'])
                    if 'HR' in stats:
                        week_team_stats[team_id]['HR'] += float(stats['HR']['value'])
                    if 'TB' in stats:
                        week_team_stats[team_id]['TB'] += float(stats['TB']['value'])
                    if 'SB-CS' in stats:
                        value = stats['SB-CS']['value']
                        if isinstance(value, str) and '-' in value:
                            sb_cs = value.split('-')
                            week_team_stats[team_id]['SB'] += float(sb_cs[0])
                            week_team_stats[team_id]['CS'] += float(sb_cs[1]) if len(sb_cs) > 1 else 0
                        else:
                            week_team_stats[team_id]['SB'] += float(value)
                    if 'H' in stats:
                        week_team_stats[team_id]['H'] += float(stats['H']['value'])
                    if 'AB' in stats:
                        week_team_stats[team_id]['AB'] += float(stats['AB']['value'])
                    if 'BB' in stats:
                        week_team_stats[team_id]['BB'] += float(stats['BB']['value'])
                    if 'K' in stats:
                        week_team_stats[team_id]['K'] += float(stats['K']['value'])
                    if 'QS' in stats:
                        week_team_stats[team_id]['QS'] += float(stats['QS']['value'])
                    if 'ER' in stats:
                        week_team_stats[team_id]['ER'] += float(stats['ER']['value'])
                    if 'OUTS' in stats:
                        week_team_stats[team_id]['OUTS'] += float(stats['OUTS']['value'])
                        week_team_stats[team_id]['IP'] += float(stats['OUTS']['value']) / 3
                    if 'P_BB' in stats:
                        week_team_stats[team_id]['P_BB'] += float(stats['P_BB']['value'])
                    if 'P_H' in stats:
                        week_team_stats[team_id]['P_H'] += float(stats['P_H']['value'])
                    if 'SVHD' in stats:
                        week_team_stats[team_id]['SVHD'] += float(stats['SVHD']['value'])
                    if 'OPS' in stats and 'AB' in stats:
                        weekly_ops = float(stats['OPS']['value'])
                        weekly_ab = float(stats['AB']['value'])
                        week_team_stats[team_id]['OPS'] += weekly_ops * weekly_ab
                        week_team_stats[team_id]['OPS_AB'] += weekly_ab
                    processed_teams.add(team_id)
                # Away team
                if box_score.away_team and box_score.away_team.team_id not in processed_teams:
                    team_id = box_score.away_team.team_id
                    stats = box_score.away_stats
                    if hasattr(box_score, 'away_lineup') and box_score.away_lineup:
                        for player in box_score.away_lineup:
                            if hasattr(player, 'slot_position') and player.slot_position == 'SP':
                                week_team_stats[team_id]['GS'] += 1
                    if 'R' in stats:
                        week_team_stats[team_id]['R'] += float(stats['R']['value'])
                    if 'HR' in stats:
                        week_team_stats[team_id]['HR'] += float(stats['HR']['value'])
                    if 'TB' in stats:
                        week_team_stats[team_id]['TB'] += float(stats['TB']['value'])
                    if 'SB-CS' in stats:
                        value = stats['SB-CS']['value']
                        if isinstance(value, str) and '-' in value:
                            sb_cs = value.split('-')
                            week_team_stats[team_id]['SB'] += float(sb_cs[0])
                            week_team_stats[team_id]['CS'] += float(sb_cs[1]) if len(sb_cs) > 1 else 0
                        else:
                            week_team_stats[team_id]['SB'] += float(value)
                    if 'H' in stats:
                        week_team_stats[team_id]['H'] += float(stats['H']['value'])
                    if 'AB' in stats:
                        week_team_stats[team_id]['AB'] += float(stats['AB']['value'])
                    if 'BB' in stats:
                        week_team_stats[team_id]['BB'] += float(stats['BB']['value'])
                    if 'K' in stats:
                        week_team_stats[team_id]['K'] += float(stats['K']['value'])
                    if 'QS' in stats:
                        week_team_stats[team_id]['QS'] += float(stats['QS']['value'])
                    if 'ER' in stats:
                        week_team_stats[team_id]['ER'] += float(stats['ER']['value'])
                    if 'OUTS' in stats:
                        week_team_stats[team_id]['OUTS'] += float(stats['OUTS']['value'])
                        week_team_stats[team_id]['IP'] += float(stats['OUTS']['value']) / 3
                    if 'P_BB' in stats:
                        week_team_stats[team_id]['P_BB'] += float(stats['P_BB']['value'])
                    if 'P_H' in stats:
                        week_team_stats[team_id]['P_H'] += float(stats['P_H']['value'])
                    if 'SVHD' in stats:
                        week_team_stats[team_id]['SVHD'] += float(stats['SVHD']['value'])
                    if 'OPS' in stats and 'AB' in stats:
                        weekly_ops = float(stats['OPS']['value'])
                        weekly_ab = float(stats['AB']['value'])
                        week_team_stats[team_id]['OPS'] += weekly_ops * weekly_ab
                        week_team_stats[team_id]['OPS_AB'] += weekly_ab
                    processed_teams.add(team_id)
        # --- Save per-week stats to DB ---
        # Calculate OPS as a rate for each team for the week
        for team_id in week_team_stats:
            ops_ab = week_team_stats[team_id]['OPS_AB']
            if ops_ab and ops_ab != 0:
                week_team_stats[team_id]['OPS'] = week_team_stats[team_id]['OPS'] / ops_ab
            else:
                week_team_stats[team_id]['OPS'] = 0
        week_df = pd.DataFrame.from_dict(week_team_stats, orient='index').copy()
        week_df['team_id'] = week_df.index
        week_df['team_name'] = week_df['team_name']
        week_df['week'] = week
        save_team_stats_by_week_to_db(week_df, week)
        # --- Accumulate into overall stats ---
        for team_id in week_team_stats:
            for stat in week_team_stats[team_id]:
                if stat in team_stats[team_id]:
                    team_stats[team_id][stat] += week_team_stats[team_id][stat]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Calculate derived stats
    df['AVG'] = df['H'] / df['AB']
    df['OPS'] = df['OPS'] / df['OPS_AB']  # Weighted average of OPS
    df['ERA'] = (df['ER'] * 9) / (df['OUTS'] / 3)
    df['WHIP'] = (df['P_BB'] + df['P_H']) / (df['OUTS'] / 3)
    df['K_BB'] = df['K'] / df['P_BB']
    df['SBN'] = df['SB'] - df['CS']  # Net steals
    
    # Select and order columns for display
    display_columns = [
        'team_name',
        'R', 'HR', 'TB', 'SBN', 'AVG', 'OPS',
        'K', 'QS', 'ERA', 'WHIP', 'K_BB', 'SVHD',
        'AB', 'IP', 'GS'  # Add new stats to display
    ]
    
    return df[display_columns]

def display_team_stats(df: pd.DataFrame):
    """
    Display team statistics in a formatted table.
    """
    # Set display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Format decimal places for rate stats
    df = df.round({
        'AVG': 3,
        'OPS': 4,
        'ERA': 2,
        'WHIP': 2,
        'K_BB': 2,
        'SLG': 4
    })
    
    # Convert counting stats to integers, but only for columns that exist
    counting_stats = ['R', 'HR', 'TB', 'SBN', 'K', 'QS', 'SVHD', 'AB', 'IP', 'GS']
    for col in counting_stats:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Sort by team name
    df_sorted = df.sort_values('team_name')
    
    # Select columns to display
    display_columns = [
        'team_name',
        'R', 'HR', 'TB', 'SBN', 'AVG', 'OPS',
        'K', 'QS', 'ERA', 'WHIP', 'K_BB', 'SVHD',
        'AB', 'IP', 'GS'
    ]
    
    print("\nTeam Statistics:")
    print(df_sorted[display_columns].to_string(index=False))

def calculate_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate points for each team based on category rankings.
    
    Args:
        df: DataFrame with team statistics
        
    Returns:
        DataFrame with team points
    """
    # Create a copy of the dataframe
    points_df = df.copy()
    
    # Define categories and their directions (1 for higher is better, -1 for lower is better)
    categories = {
        'R': 1,
        'HR': 1,
        'TB': 1,
        'SBN': 1,
        'AVG': 1,
        'OPS': 1,
        'K': 1,
        'QS': 1,
        'ERA': -1,
        'WHIP': -1,
        'K_BB': 1,
        'SVHD': 1
    }
    
    # Calculate points for each category
    for category, direction in categories.items():
        # Sort values based on direction
        sorted_values = points_df[category].sort_values(ascending=(direction == -1))
        
        # Group by value to handle ties
        value_groups = sorted_values.groupby(sorted_values)
        
        # Calculate points for each group
        points = pd.Series(index=sorted_values.index, dtype=float)
        current_rank = 1
        for value, group in value_groups:
            # Calculate median rank for this group
            group_size = len(group)
            median_rank = current_rank + (group_size - 1) / 2
            # Assign points based on median rank (10 points for 1st, 9 for 2nd, etc.)
            points[group.index] = 11 - median_rank
            current_rank += group_size
        
        points_df[f'{category}_points'] = points
    
    # Calculate points for display-only categories (not included in total)
    display_categories = {
        'AB': 1,
        'IP': 1,
        'GS': 1
    }
    
    for category, direction in display_categories.items():
        if category in points_df.columns:
            # Sort values based on direction
            sorted_values = points_df[category].sort_values(ascending=(direction == -1))
            
            # Group by value to handle ties
            value_groups = sorted_values.groupby(sorted_values)
            
            # Calculate points for each group
            points = pd.Series(index=sorted_values.index, dtype=float)
            current_rank = 1
            for value, group in value_groups:
                # Calculate median rank for this group
                group_size = len(group)
                median_rank = current_rank + (group_size - 1) / 2
                # Assign points based on median rank (10 points for 1st, 9 for 2nd, etc.)
                points[group.index] = 11 - median_rank
                current_rank += group_size
            
            points_df[f'{category}_points'] = points
    
    # Calculate total points (excluding display-only categories)
    points_df['Total Points'] = points_df[[f'{cat}_points' for cat in categories.keys()]].sum(axis=1)
    
    # Sort by total points
    points_df = points_df.sort_values('Total Points', ascending=False)
    
    # Select columns to display - include all points columns
    display_columns = ['team_name', 'Total Points'] + [f'{cat}_points' for cat in categories.keys()] + [f'{cat}_points' for cat in display_categories.keys()]
    
    return points_df[display_columns]

def display_points_table(df: pd.DataFrame):
    """
    Display points table in a formatted way.
    
    Args:
        df: DataFrame with points
    """
    # Set display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Format the display
    print("\nPoints Table:")
    print(df.to_string(index=False))
    print("\n")

def save_team_stats_to_db(df: pd.DataFrame):
    """
    Save team statistics to the SQLite database.
    
    Args:
        df: DataFrame with team statistics
    """
    try:
        db_url = os.environ['DATABASE_URL']
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        engine = create_engine(db_url)
        df.to_sql('team_stats', engine, if_exists='replace', index=False)
        print(f"Successfully saved team stats to database")
    except Exception as e:
        print(f"Error saving team stats to database: {e}")
        raise

def save_team_stats_by_week_to_db(df: pd.DataFrame, week: int):
    """
    Save team statistics for a single week to the SQLite database.
    Args:
        df: DataFrame with team statistics for one week
        week: The matchup week number
    """
    try:
        db_url = os.environ['DATABASE_URL']
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        engine = create_engine(db_url)
        df['week'] = week
        df.columns = [col.lower() for col in df.columns]
        df.to_sql('team_stats_by_week', engine, if_exists='append', index=False)
        print(f"Successfully saved team stats for week {week} to database")
    except Exception as e:
        print(f"Error saving team stats for week {week} to database: {e}")
        raise

def main():
    """
    Main function to run the team stats aggregator and display results.
    """
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment variables
    league_id = int(os.getenv('LEAGUE_ID'))
    year = int(os.getenv('YEAR'))
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('SWID')
    
    # Initialize league with credentials
    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    from espn_api.baseball.box_score import H2HCategoryBoxScore
    league._box_score_class = H2HCategoryBoxScore
    
    # Aggregate team stats
    df = aggregate_team_stats(league)
    
    # Display team stats
    print("\nTeam Statistics:")
    display_team_stats(df)
    
    # Calculate and display points
    points_df = calculate_points(df)
    display_points_table(points_df)
    
    # Save to database
    save_team_stats_to_db(df)

if __name__ == "__main__":
    main() 