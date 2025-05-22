import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_csv_files():
    """Initialize the CSV files with proper structure."""
    try:
        # Create empty dataframes with proper columns
        hitters_df = pd.DataFrame(columns=[
            'player_id', 'name', 'team', 'R', 'HR', 'H', 'AB', 'PA', 'G',
            'AVG', 'OPS', 'wOBA', 'wRC_plus', 'WAR', 'TB', 'SBN',
            'proj_R', 'proj_HR', 'proj_H', 'proj_AB', 'proj_PA', 'proj_G',
            'proj_AVG', 'proj_OPS', 'proj_wOBA', 'proj_wRC_plus', 'proj_WAR',
            'proj_TB', 'proj_SBN', 'proj_R_z', 'proj_HR_z', 'proj_TB_z',
            'proj_SBN_z', 'proj_AVG_z', 'proj_OPS_z', 'proj_PR',
            'blended_R', 'blended_HR', 'blended_TB', 'blended_SBN',
            'blended_AVG', 'blended_OPS', 'blended_R_z', 'blended_HR_z',
            'blended_TB_z', 'blended_SBN_z', 'blended_AVG_z', 'blended_OPS_z',
            'blended_PR'
        ])
        
        pitchers_df = pd.DataFrame(columns=[
            'player_id', 'name', 'team', 'W', 'L', 'SV', 'K', 'ERA', 'WHIP',
            'IP', 'G', 'proj_W', 'proj_L', 'proj_SV', 'proj_K', 'proj_ERA',
            'proj_WHIP', 'proj_IP', 'proj_G', 'proj_K_z', 'proj_QS_z',
            'proj_ERA_z', 'proj_WHIP_z', 'proj_K_BB_z', 'proj_SVHD_z',
            'proj_PR', 'blended_K', 'blended_QS', 'blended_ERA',
            'blended_WHIP', 'blended_K_BB', 'blended_SVHD', 'blended_K_z',
            'blended_QS_z', 'blended_ERA_z', 'blended_WHIP_z',
            'blended_K_BB_z', 'blended_SVHD_z', 'blended_PR'
        ])
        
        # Save to CSV files
        hitters_df.to_csv('data/hitter_stats.csv', index=False)
        pitchers_df.to_csv('data/pitcher_stats.csv', index=False)
        
        logger.info("CSV files initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing CSV files: {e}")
        raise

if __name__ == "__main__":
    init_csv_files() 