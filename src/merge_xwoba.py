import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_id(val):
    # Convert to string and remove trailing .0 if present
    if pd.isna(val):
        return ''
    val_str = str(val)
    if val_str.endswith('.0'):
        return val_str[:-2]
    return val_str

def merge_xwoba():
    # Read the files
    logger.info("Reading hitter_stats.csv...")
    hitter_stats = pd.read_csv('data/hitter_stats.csv', dtype={'mlbamid': 'str'})
    
    logger.info("Reading baseball_stats_2025.csv...")
    baseball_stats = pd.read_csv('batter_stats_2025.csv', dtype={'mlb_ID': 'str', 'IDfg': 'str'})
    
    # Standardize ID columns
    hitter_stats['mlbamid'] = hitter_stats['mlbamid'].apply(clean_id)
    baseball_stats['mlb_ID'] = baseball_stats['mlb_ID'].apply(clean_id)
    
    hitter_stats['fangraphs_id'] = hitter_stats['fangraphs_id'].apply(clean_id)
    baseball_stats['IDfg'] = baseball_stats['IDfg'].apply(clean_id)
    
    # Rename xwoba to xwOBA_percentile in baseball_stats to avoid duplicate columns
    baseball_stats = baseball_stats.rename(columns={'xwoba': 'xwOBA_percentile'})
    
    # Log initial shapes
    logger.info(f"hitter_stats shape: {hitter_stats.shape}")
    logger.info(f"baseball_stats shape: {baseball_stats.shape}")
    
    # Verify xwOBA and xwOBA_percentile columns exist
    if 'xwOBA' not in baseball_stats.columns or 'xwOBA_percentile' not in baseball_stats.columns:
        logger.error("xwOBA or xwOBA_percentile column not found in baseball_stats_2025.csv")
        return
    
    logger.info("\nSample of hitter_stats IDs:")
    logger.info(hitter_stats[['name', 'mlbamid', 'fangraphs_id']].head().to_string())
    
    logger.info("\nSample of baseball_stats IDs:")
    logger.info(baseball_stats[['Name', 'mlb_ID', 'IDfg']].head().to_string())
    
    # First try matching on mlbamid
    logger.info("\nAttempting to match on mlbamid...")
    merged_stats = pd.merge(
        hitter_stats,
        baseball_stats[['mlb_ID', 'xwOBA', 'xwOBA_percentile']].drop_duplicates(subset=['mlb_ID']),
        left_on='mlbamid',
        right_on='mlb_ID',
        how='left'
    )
    
    # If columns don't exist after merge, create them
    if 'xwOBA' not in merged_stats.columns:
        merged_stats['xwOBA'] = pd.NA
    if 'xwOBA_percentile' not in merged_stats.columns:
        merged_stats['xwOBA_percentile'] = pd.NA
    
    # Find unmatched rows
    unmatched = merged_stats[merged_stats['xwOBA'].isna() & merged_stats['xwOBA_percentile'].isna()]
    matched = merged_stats[~(merged_stats['xwOBA'].isna() & merged_stats['xwOBA_percentile'].isna())]
    
    logger.info(f"\nMatched on mlbamid: {len(matched)}")
    logger.info(f"Unmatched on mlbamid: {len(unmatched)}")
    
    if len(unmatched) > 0:
        logger.info(f"\nTrying fangraphs_id match...")
        unmatched = pd.merge(
            unmatched.drop(columns=['mlb_ID', 'xwOBA', 'xwOBA_percentile']),
            baseball_stats[['IDfg', 'xwOBA', 'xwOBA_percentile']].drop_duplicates(subset=['IDfg']),
            left_on='fangraphs_id',
            right_on='IDfg',
            how='left'
        )
        # If columns don't exist after merge, create them
        if 'xwOBA' not in unmatched.columns:
            unmatched['xwOBA'] = pd.NA
        if 'xwOBA_percentile' not in unmatched.columns:
            unmatched['xwOBA_percentile'] = pd.NA
        # Combine matched and unmatched
        merged_stats = pd.concat([matched, unmatched], ignore_index=True)
    
    # Clean up helper columns
    merged_stats = merged_stats.drop(columns=['mlb_ID', 'IDfg'], errors='ignore')
    
    # Clean up any duplicate xwoba columns
    for col in ['xwoba_x', 'xwoba_y', 'xwoba']:
        if col in merged_stats.columns:
            merged_stats = merged_stats.drop(columns=[col])
    
    # Consolidate xwOBA columns if needed
    if 'xwOBA_y' in merged_stats.columns:
        merged_stats['xwOBA'] = merged_stats['xwOBA'].combine_first(merged_stats['xwOBA_y'])
        merged_stats = merged_stats.drop(columns=['xwOBA_y'])
    if 'xwOBA_x' in merged_stats.columns:
        merged_stats['xwOBA'] = merged_stats['xwOBA_x'].combine_first(merged_stats['xwOBA'])
        merged_stats = merged_stats.drop(columns=['xwOBA_x'])
    if 'xwOBA_percentile_y' in merged_stats.columns:
        merged_stats['xwOBA_percentile'] = merged_stats['xwOBA_percentile'].combine_first(merged_stats['xwOBA_percentile_y'])
        merged_stats = merged_stats.drop(columns=['xwOBA_percentile_y'])
    if 'xwOBA_percentile_x' in merged_stats.columns:
        merged_stats['xwOBA_percentile'] = merged_stats['xwOBA_percentile_x'].combine_first(merged_stats['xwOBA_percentile'])
        merged_stats = merged_stats.drop(columns=['xwOBA_percentile_x'])
    
    # Ensure no duplicates in final dataset
    merged_stats = merged_stats.drop_duplicates(subset=['player_id'])
    
    # Log matching statistics
    total_players = len(merged_stats)
    matched_players = len(merged_stats[~merged_stats['xwOBA'].isna()])
    logger.info(f"\nMatching statistics:")
    logger.info(f"Total players: {total_players}")
    logger.info(f"Players with xwOBA: {matched_players}")
    logger.info(f"Players without xwOBA: {total_players - matched_players}")
    
    # Save the updated stats
    logger.info("Saving updated hitter stats...")
    merged_stats.to_csv('data/hitter_stats.csv', index=False)
    
    logger.info("Done!")

if __name__ == "__main__":
    merge_xwoba() 