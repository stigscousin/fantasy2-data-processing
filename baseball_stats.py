from pybaseball import statcast_batter_percentile_ranks, statcast_batter_expected_stats, fg_batting_data, playerid_lookup
import pandas as pd
from curl_cffi import requests
from io import StringIO
import numpy as np

# Load SFBB Player ID Map
print("Loading SFBB Player ID Map...")
try:
    id_map = pd.read_csv('SFBB Player ID Map - PLAYERIDMAP.csv')
    print("Successfully loaded SFBB Player ID Map")
except Exception as e:
    print(f"Error loading SFBB Player ID Map: {str(e)}")
    exit(1)

# Get Baseball Reference WAR data
print("Fetching Baseball Reference WAR data...")
try:
    url = "https://www.baseball-reference.com/data/war_daily_bat.txt"
    response = requests.get(url, impersonate="chrome")
    response.raise_for_status()
    bbref_df = pd.read_csv(StringIO(response.text))
    print("Successfully loaded Baseball Reference data")
except Exception as e:
    print(f"Error fetching Baseball Reference data: {str(e)}")
    exit(1)

# Get Fangraphs data for 2025 - all batters (qual=0)
print("Fetching Fangraphs data...")
fangraphs_data = fg_batting_data(2025, qual=0)
print(f"Fangraphs data shape: {fangraphs_data.shape}")

# Get percentile ranks for all qualified batters in 2025
print("Fetching Statcast percentile data...")
percentile_data = statcast_batter_percentile_ranks(2025)
print(f"Percentile data shape: {percentile_data.shape}")

# Get expected statistics for all qualified batters in 2025
print("Fetching Statcast expected stats...")
expected_data = statcast_batter_expected_stats(2025)
print(f"Expected data shape: {expected_data.shape}")

# Merge percentile and expected data
statcast_data = pd.merge(
    percentile_data,
    expected_data,
    on='player_id',
    suffixes=('_percentile', '_expected')
)
print(f"Statcast data shape: {statcast_data.shape}")

# Filter Baseball Reference data for 2025
bbref_2025 = bbref_df[bbref_df['year_ID'] == 2025].copy()
print(f"Baseball Reference 2025 data shape: {bbref_2025.shape}")

# Clean and prepare ID columns
id_map = id_map[['IDFANGRAPHS', 'MLBID', 'BREFID']].copy()

# Convert IDs to strings and handle non-numeric values
id_map['IDFANGRAPHS'] = id_map['IDFANGRAPHS'].astype(str)
id_map['BREFID'] = id_map['BREFID'].astype(str)

# Clean up MLB IDs - handle NaN values and convert to integers
id_map['MLBID'] = id_map['MLBID'].fillna('').astype(str)
id_map = id_map[id_map['MLBID'] != '']  # Remove rows with empty MLB IDs
id_map['MLBID'] = id_map['MLBID'].astype(float).astype(int).astype(str)

# Remove rows where any ID is empty or NaN
id_map = id_map.replace('', np.nan)
id_map = id_map.dropna(subset=['IDFANGRAPHS', 'BREFID'])

# Prepare data sources for merging
fangraphs_data['IDfg'] = fangraphs_data['IDfg'].astype(str)
statcast_data['player_id'] = statcast_data['player_id'].astype(str)
bbref_2025['player_ID'] = bbref_2025['player_ID'].astype(str)

# First merge: Fangraphs with Baseball Reference using SFBB IDs
print("\nMerging Fangraphs with Baseball Reference using SFBB IDs...")
id_matched = pd.merge(
    fangraphs_data,
    id_map[['IDFANGRAPHS', 'BREFID']],
    left_on='IDfg',
    right_on='IDFANGRAPHS',
    how='left'
)

id_matched = pd.merge(
    id_matched,
    bbref_2025,
    left_on='BREFID',
    right_on='player_ID',
    how='left'
)
print(f"After ID-based merge: {id_matched.shape}")

# Find players missing Baseball Reference data
missing_bbref = id_matched[id_matched['player_ID'].isna()].copy()
print(f"\nFound {len(missing_bbref)} players missing Baseball Reference data")

# Prepare for name-based matching
if len(missing_bbref) > 0:
    print("Attempting name-based matching for missing players...")
    # Clean names for matching
    missing_bbref['clean_name'] = missing_bbref['Name'].str.lower().str.replace('.', '').str.replace(' ', '')
    bbref_2025['clean_name'] = bbref_2025['name_common'].str.lower().str.replace('.', '').str.replace(' ', '')
    
    # Merge on clean names
    name_matched = pd.merge(
        missing_bbref.drop(columns=[col for col in missing_bbref.columns if col in bbref_2025.columns and col != 'clean_name']),
        bbref_2025,
        on='clean_name',
        how='left'
    )
    
    # Update the main dataset with name-matched data
    if not name_matched.empty:
        print(f"Found {len(name_matched[~name_matched['player_ID'].isna()])} name-based matches")
        # Drop the original unmatched rows
        id_matched = id_matched[~id_matched['IDfg'].isin(missing_bbref['IDfg'])]
        # Add the name-matched rows
        merged_data = pd.concat([id_matched, name_matched], ignore_index=True)
    else:
        merged_data = id_matched
else:
    merged_data = id_matched

# Merge with Statcast data using SFBB IDs
print("\nMerging with Statcast data...")
merged_data = pd.merge(
    merged_data,
    id_map[['IDFANGRAPHS', 'MLBID']],
    left_on='IDfg',
    right_on='IDFANGRAPHS',
    how='left'
)

merged_data = pd.merge(
    merged_data,
    statcast_data,
    left_on='MLBID',
    right_on='player_id',
    how='left'
)
print(f"After Statcast merge: {merged_data.shape}")

# Clean up the final dataset
# Drop helper columns that exist
columns_to_drop = ['clean_name', 'IDFANGRAPHS', 'BREFID', 'MLBID']
existing_columns = [col for col in columns_to_drop if col in merged_data.columns]
if existing_columns:
    merged_data = merged_data.drop(columns=existing_columns)

# Print matching statistics
print("\nMatching statistics:")
print(f"Total players in Fangraphs: {len(fangraphs_data)}")
print(f"Players with ID matches in all three sources: {len(merged_data[~merged_data['player_ID'].isna() & ~merged_data['player_id'].isna()])}")
print(f"Players missing Baseball Reference data: {len(merged_data[merged_data['player_ID'].isna()])}")
print(f"Players missing Statcast data: {len(merged_data[merged_data['player_id'].isna()])}")

# Save to CSV
output_file = 'batter_stats_2025.csv'
merged_data.to_csv(output_file, index=False)
print(f"\nCombined stats saved to {output_file} with {len(merged_data)} rows") 