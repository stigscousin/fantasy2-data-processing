import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define roster positions and their order of priority
ROSTER_POSITIONS = ['C', '1B', '2B', 'SS', '3B', 'OF', 'OF', 'OF', 'UTIL']

# All stat columns are now lowercase (e.g., 'proj_pr', not 'proj_PR')

def get_eligible_players_for_position(players: pd.DataFrame, position: str) -> pd.DataFrame:
    """Get players eligible for a specific position."""
    if position == 'UTIL':
        # For UTIL, any hitter is eligible
        return players
    return players[players['eligible_positions'].str.contains(position, na=False)]

def select_optimal_roster(players: pd.DataFrame) -> pd.DataFrame:
    """Select the optimal roster for a team based on position eligibility and PR."""
    team_players = players.copy()
    selected_players = []
    remaining_positions = ROSTER_POSITIONS.copy()
    
    # Log initial state
    logger.info(f"Selecting optimal roster from {len(team_players)} available players")
    
    # First, fill required positions in order
    for position in ROSTER_POSITIONS[:-1]:  # Exclude UTIL
        eligible = get_eligible_players_for_position(team_players, position)
        if not eligible.empty:
            # Get the player with highest projected PR
            best_player = eligible.nlargest(1, 'proj_pr').iloc[0]
            selected_players.append(best_player)
            team_players = team_players[team_players['player_id'] != best_player['player_id']]
            remaining_positions.remove(position)
            logger.info(f"Selected {best_player['name']} for {position} (PR: {best_player['proj_pr']:.2f})")
    
    # Fill UTIL with best remaining player
    if team_players.empty:
        logger.info("No players left for UTIL position")
    else:
        best_util = team_players.nlargest(1, 'proj_pr').iloc[0]
        selected_players.append(best_util)
        logger.info(f"Selected {best_util['name']} for UTIL (PR: {best_util['proj_pr']:.2f})")
    
    return pd.DataFrame(selected_players) 