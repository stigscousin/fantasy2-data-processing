import os
from dotenv import load_dotenv
from espn_api.baseball import League
import pandas as pd
import sqlite3
from typing import Dict, List, Union
from datetime import datetime
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import numpy as np
from scipy import stats
from constants import POSITION_MAP, HITTER_POSITIONS, PITCHER_POSITIONS, OF_POSITIONS
import psycopg2
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FantasyStatsFetcher:
    def __init__(self, league_id: int, year: int):
        """Initialize the stats fetcher with league credentials"""
        self.league_id = league_id
        self.year = year
        self.espn_s2 = os.getenv('ESPN_S2')
        self.swid = os.getenv('SWID')
        
        # Initialize the league connection
        self.league = League(
            league_id=league_id,
            year=year,
            espn_s2=self.espn_s2,
            swid=self.swid
        )
        
        # Initialize Postgres database
        self.db_url = os.environ['DATABASE_URL']
        if self.db_url.startswith('postgres://'):
            self.db_url = self.db_url.replace('postgres://', 'postgresql://', 1)
        self.engine = create_engine(self.db_url)
        self._init_db()
        
        # Load SFBB Player ID Map
        self.id_map = pd.read_csv('SFBB Player ID Map - PLAYERIDMAP.csv')
        self.id_map['ESPNID'] = self.id_map['ESPNID'].astype(str).str.replace('\.0$', '', regex=True)
        self.id_map['MLBID'] = self.id_map['MLBID'].astype(str).str.replace('\.0$', '', regex=True)
        self.id_map['IDFANGRAPHS'] = self.id_map['IDFANGRAPHS'].astype(str).str.replace('\.0$', '', regex=True)
        # Remove rows where all IDs are empty
        self.id_map = self.id_map[~(self.id_map['ESPNID'].isna() & self.id_map['MLBID'].isna() & self.id_map['IDFANGRAPHS'].isna())]
        logger.info(f"Loaded ID map with {len(self.id_map)} entries")
        logger.info(f"ID map columns: {self.id_map.columns.tolist()}")
    
    def _init_db(self):
        """Initialize Postgres database with required tables"""
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()
        
        # Drop pitchers table if it exists to ensure schema is up to date
        cursor.execute('DROP TABLE IF EXISTS pitchers')
        # Create pitchers table with all columns needed for projections
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitchers (
            name TEXT,
            position TEXT,
            eligible_positions TEXT,
            team TEXT,
            player_id INTEGER,
            fangraphs_id TEXT,
            mlbamid TEXT,
            is_free_agent INTEGER,
            K REAL,
            QS REAL,
            ERA REAL,
            WHIP REAL,
            K_BB REAL,
            SVHD REAL,
            G REAL,
            IP REAL,
            ER REAL,
            BB REAL,
            K_BB_diff REAL,
            team_kbb_contribution REAL,
            ERA_diff REAL,
            WHIP_diff REAL,
            K_z REAL,
            QS_z REAL,
            SVHD_z REAL,
            ERA_z REAL,
            WHIP_z REAL,
            K_BB_z REAL,
            current_PR REAL,
            blended_K REAL,
            blended_QS REAL,
            blended_ERA REAL,
            blended_WHIP REAL,
            blended_K_BB REAL,
            blended_SVHD REAL,
            blended_PR REAL,
            blended_ERA_diff REAL,
            blended_WHIP_diff REAL,
            blended_K_BB_diff REAL,
            blended_K_z REAL,
            blended_QS_z REAL,
            blended_ERA_z REAL,
            blended_WHIP_z REAL,
            blended_K_BB_z REAL,
            blended_SVHD_z REAL,
            proj_G REAL,
            proj_GS REAL,
            proj_IP REAL,
            proj_K REAL,
            proj_BB REAL,
            proj_HR REAL,
            proj_ERA REAL,
            proj_WHIP REAL,
            proj_W REAL,
            proj_L REAL,
            proj_SV REAL,
            proj_HLD REAL,
            proj_SVHD REAL,
            proj_K_BB REAL,
            proj_QS REAL,
            proj_ERA_diff REAL,
            proj_WHIP_diff REAL,
            proj_K_z REAL,
            proj_QS_z REAL,
            proj_ERA_z REAL,
            proj_WHIP_z REAL,
            proj_K_BB_z REAL,
            proj_SVHD_z REAL,
            proj_PR REAL,
            proj_team_kbb_contribution REAL,
            proj_K_BB_diff REAL
        )
        ''')
        
        # Create hitters table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hitters (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            position TEXT NOT NULL,
            eligible_positions TEXT NOT NULL,
            team TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            fangraphs_id TEXT,
            mlbamid TEXT,
            is_free_agent INTEGER NOT NULL,
            R REAL,
            HR REAL,
            TB REAL,
            SBN REAL,
            AVG REAL,
            OPS REAL,
            H REAL,
            AB REAL,
            PA REAL,
            G REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _save_to_sqlite(self, stats: Dict[str, List[Dict]]):
        """Save stats to Postgres database"""
        # Save hitters
        hitters_df = pd.DataFrame(stats['hitters'])
        if not hitters_df.empty:
            if 'PR' in hitters_df.columns:
                hitters_df = hitters_df.rename(columns={'PR': 'current_PR'})
            hitters_df.columns = hitters_df.columns.str.lower()
            if 'is_free_agent' in hitters_df.columns:
                hitters_df['is_free_agent'] = hitters_df['is_free_agent'].astype(int)
            hitters_df.to_sql('hitters', self.engine, if_exists='replace', index=False)
            hitters_df.to_csv('data/hitter_stats.csv', index=False)
        
        # Save pitchers
        pitchers_df = pd.DataFrame(stats['pitchers'])
        if not pitchers_df.empty:
            if 'PR' in pitchers_df.columns:
                pitchers_df = pitchers_df.rename(columns={'PR': 'current_PR'})
            pitchers_df.columns = pitchers_df.columns.str.lower()
            if 'is_free_agent' in pitchers_df.columns:
                pitchers_df['is_free_agent'] = pitchers_df['is_free_agent'].astype(int)
            # Truncate the table to keep schema, then append
            with self.engine.begin() as conn:
                conn.execute(text('TRUNCATE TABLE pitchers'))
            pitchers_df.to_sql('pitchers', self.engine, if_exists='append', index=False)
            pitchers_df.to_csv('data/pitcher_stats.csv', index=False)
        
        # Log some statistics about the ID mapping
        if not hitters_df.empty:
            hitters_with_fg = hitters_df[hitters_df['fangraphs_id'].notna()]
            hitters_with_mlb = hitters_df[hitters_df['mlbamid'].notna()]
            logger.info(f"Found FanGraphs IDs for {len(hitters_with_fg)} out of {len(hitters_df)} hitters")
            logger.info(f"Found MLBAMIDs for {len(hitters_with_mlb)} out of {len(hitters_df)} hitters")
        
        if not pitchers_df.empty:
            pitchers_with_fg = pitchers_df[pitchers_df['fangraphs_id'].notna()]
            pitchers_with_mlb = pitchers_df[pitchers_df['mlbamid'].notna()]
            logger.info(f"Found FanGraphs IDs for {len(pitchers_with_fg)} out of {len(pitchers_df)} pitchers")
            logger.info(f"Found MLBAMIDs for {len(pitchers_with_mlb)} out of {len(pitchers_df)} pitchers")
    
    def _get_eligible_positions(self, player) -> List[str]:
        """Get list of eligible positions for a player directly from ESPN"""
        if not hasattr(player, 'eligibleSlots'):
            return []
        
        positions = []
        has_of = False
        for pos_id in player.eligibleSlots:
            pos = POSITION_MAP.get(pos_id, str(pos_id))
            # Only add valid positions (not UTIL, IL, etc)
            if pos in HITTER_POSITIONS or pos in PITCHER_POSITIONS:
                if pos in OF_POSITIONS:
                    if not has_of:
                        positions.append('OF')
                        has_of = True
                else:
                    positions.append(pos)
        
        return positions
    
    def _is_pitcher(self, player) -> bool:
        """Determine if a player is a pitcher based on their eligible positions"""
        if not hasattr(player, 'eligibleSlots'):
            return False
        return any(POSITION_MAP.get(pos, str(pos)) in PITCHER_POSITIONS for pos in player.eligibleSlots)
    
    def _is_hitter(self, player) -> bool:
        """Determine if a player is a hitter based on their eligible positions"""
        if not hasattr(player, 'eligibleSlots'):
            return False
        positions = set()
        for pos_id in player.eligibleSlots:
            pos = POSITION_MAP.get(pos_id, str(pos_id))
            if pos in OF_POSITIONS:
                positions.add('OF')
            elif pos in HITTER_POSITIONS:
                positions.add(pos)
        return len(positions) > 0
    
    def _process_player_stats(self, player, team_name: str, is_free_agent: bool) -> Union[Dict, None]:
        """Process a single player's stats and return the appropriate data structure"""
        try:
            # Get FanGraphs ID and MLBAMID from the ID map
            fangraphs_id = None
            mlbamid = None
            player_espn_id = str(player.playerId)
            if player_espn_id in self.id_map['ESPNID'].values:
                player_row = self.id_map[self.id_map['ESPNID'] == player_espn_id].iloc[0]
                fangraphs_id = player_row['IDFANGRAPHS'] if pd.notna(player_row['IDFANGRAPHS']) else None
                mlbamid = player_row['MLBID'] if pd.notna(player_row['MLBID']) else None
                logger.debug(f"Found IDs for {player.name}: ESPN={player_espn_id}, FanGraphs={fangraphs_id}, MLBAM={mlbamid}")
            else:
                logger.debug(f"No ID map entry found for {player.name} (ESPN ID: {player_espn_id})")
            
            # Get eligible positions
            eligible_positions = self._get_eligible_positions(player)
            if not eligible_positions:
                logger.warning(f"Player {player.name} has no eligible positions")
                return None
                
            # Deduplicate positions before setting primary position
            unique_positions = list(dict.fromkeys(eligible_positions))
            primary_position = unique_positions[0]
            
            player_data = {
                'name': player.name,
                'position': primary_position,
                'eligible_positions': ','.join(unique_positions),
                'team': team_name,
                'player_id': player.playerId,
                'fangraphs_id': fangraphs_id if fangraphs_id and fangraphs_id != '' else None,
                'mlbamid': mlbamid if mlbamid and mlbamid != '' else None,
                'is_free_agent': is_free_agent
            }
            
            if 0 in player.stats:  # 0 represents current season
                stats = player.stats[0]
                if 'breakdown' in stats:
                    current_stats = stats['breakdown']
                    
                    # Debug logging to see available stats
                    if self._is_pitcher(player):
                        logger.debug(f"Available stats for pitcher {player.name}: {list(current_stats.keys())}")
                    
                    if self._is_hitter(player):
                        # Only include hitters with at least one plate appearance
                        if current_stats.get('PA', 0) == 0:
                            return None
                            
                        player_data.update({
                            'R': current_stats.get('R', 0),
                            'HR': current_stats.get('HR', 0),
                            'TB': current_stats.get('TB', 0),
                            'SBN': current_stats.get('SB', 0) - current_stats.get('CS', 0),  # Calculate net stolen bases
                            'AVG': current_stats.get('AVG', 0),
                            'OPS': current_stats.get('OPS', 0),
                            'H': current_stats.get('H', 0),
                            'AB': current_stats.get('AB', 0),
                            'PA': current_stats.get('PA', 0),
                            'G': current_stats.get('G', 0)
                        })
                        return {'type': 'hitter', 'data': player_data}
                    elif self._is_pitcher(player):
                        # Only include pitchers with at least one inning pitched
                        if current_stats.get('OUTS', 0) == 0:
                            return None
                            
                        player_data.update({
                            'K': current_stats.get('K', 0),
                            'QS': current_stats.get('QS', 0),
                            'ERA': current_stats.get('ERA', 0),
                            'WHIP': current_stats.get('WHIP', 0),
                            'K_BB': float(current_stats.get('K', 0)) / float(current_stats.get('P_BB', 1)) if current_stats.get('P_BB', 0) > 0 else float(current_stats.get('K', 0)),  # Calculate K/BB directly from K and BB totals, or just K if BB is 0
                            'SVHD': current_stats.get('SVHD', 0),
                            'G': current_stats.get('GP', 0),  # Use GP for pitchers
                            'IP': current_stats.get('OUTS', 0) / 3,  # Convert OUTS to IP
                            'ER': current_stats.get('ER', 0),
                            'BB': current_stats.get('P_BB', 0)  # Add walks
                        })
                        return {'type': 'pitcher', 'data': player_data}
            
            return None
        except Exception as e:
            logger.error(f"Error processing player {player.name}: {str(e)}")
            return None
    
    def _get_espn_request_session(self):
        """Create a requests session with retries and proper headers"""
        session = requests.Session()
        
        # Configure retries
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Set headers
        session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Cookie': f'SWID={self.swid}; espn_s2={self.espn_s2}',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': f'https://fantasy.espn.com/baseball/team?leagueId={self.league_id}&seasonId={self.year}'
        })
        
        return session
    
    def _calculate_pr_values(self, stats_df: pd.DataFrame, is_hitter: bool) -> pd.DataFrame:
        """Calculate PR (Player Rating) values for each player based on z-scores of their stats.
        
        For hitters:
        - R, HR, TB, SBN: Direct z-score
        - AVG, OPS: Weighted by ABs (difference from league average * ABs, then z-score)
        
        For pitchers:
        - K, QS, SVHD: Direct z-score
        - ERA, WHIP, K/BB: Weighted by IP (difference from league average * IP, then z-score)
        """
        df = stats_df.copy()
        
        if is_hitter:
            # Calculate weighted averages for rate stats
            league_avg = np.average(df['AVG'], weights=df['AB'])
            league_ops = np.average(df['OPS'], weights=df['AB'])
            
            # Calculate weighted differences (normalized by AB)
            df['AVG_diff'] = (df['AVG'] - league_avg) * df['AB'] / df['AB'].mean()
            df['OPS_diff'] = (df['OPS'] - league_ops) * df['AB'] / df['AB'].mean()
            
            # Calculate z-scores for counting stats
            for stat in ['R', 'HR', 'TB', 'SBN']:
                df[f'{stat}_z'] = stats.zscore(df[stat])
            
            # Calculate z-scores for weighted rate stats
            df['AVG_z'] = stats.zscore(df['AVG_diff'])
            df['OPS_z'] = stats.zscore(df['OPS_diff'])
            
            # Calculate total PR
            df['PR'] = df[['R_z', 'HR_z', 'TB_z', 'SBN_z', 'AVG_z', 'OPS_z']].sum(axis=1)
            
        else:  # Pitchers
            # Calculate weighted averages for rate stats
            league_era = np.average(df['ERA'], weights=df['IP'])
            league_whip = np.average(df['WHIP'], weights=df['IP'])
            
            # Calculate league totals and average K/BB ratio
            total_league_k = np.sum(df['K'])
            total_league_bb = np.sum(df['BB'])
            league_kbb = total_league_k / total_league_bb
            
            # Calculate each pitcher's contribution to team K/BB ratio
            # This measures how much they would improve a team's ratio if added to league average totals
            df['team_kbb_contribution'] = (total_league_k + df['K']) / (total_league_bb + df['BB']) - league_kbb
            
            # Calculate weighted differences (normalized by IP)
            df['ERA_diff'] = (league_era - df['ERA']) * df['IP'] / df['IP'].mean()
            df['WHIP_diff'] = (league_whip - df['WHIP']) * df['IP'] / df['IP'].mean()
            
            # Calculate z-scores for counting stats
            for stat in ['K', 'QS', 'SVHD']:
                df[f'{stat}_z'] = stats.zscore(df[stat])
            
            # Calculate z-scores for weighted rate stats
            df['ERA_z'] = stats.zscore(df['ERA_diff'])
            df['WHIP_z'] = stats.zscore(df['WHIP_diff'])
            
            # Calculate z-score for team K/BB contribution
            df['K_BB_z'] = stats.zscore(df['team_kbb_contribution'])
            
            # Calculate total PR
            df['PR'] = df[['K_z', 'QS_z', 'ERA_z', 'WHIP_z', 'K_BB_z', 'SVHD_z']].sum(axis=1)
        
        # Replace infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        return df

    def get_player_stats(self) -> Dict[str, List[Dict]]:
        """Fetch all player stats from the league"""
        # Initialize stats containers
        hitter_stats = []
        pitcher_stats = []
        processed_players = set()  # Use a set to track unique player IDs
        
        # Process team rosters first
        logger.info("Processing team rosters...")
        for team in self.league.teams:
            logger.info(f"Processing team: {team.team_name}")
            for player in team.roster:
                result = self._process_player_stats(player, team.team_name, False)
                if result:
                    player_key = f"{player.playerId}_{result['type']}"  # Use playerId and type as key
                    if player_key not in processed_players:
                        if result['type'] == 'hitter':
                            hitter_stats.append(result['data'])
                        else:
                            pitcher_stats.append(result['data'])
                        processed_players.add(player_key)
        
        # Fetch free agents with larger size parameter
        logger.info("Fetching free agents...")
        try:
            free_agents = self.league.free_agents(size=1000)  # Get up to 1000 free agents
            logger.info(f"Found {len(free_agents)} free agents")
            
            for player in free_agents:
                result = self._process_player_stats(player, 'Free Agent', True)
                if result:
                    player_key = f"{player.playerId}_{result['type']}"  # Use playerId and type as key
                    if player_key not in processed_players:
                        if result['type'] == 'hitter':
                            hitter_stats.append(result['data'])
                        else:
                            pitcher_stats.append(result['data'])
                        processed_players.add(player_key)
                    
        except Exception as e:
            logger.error(f"Error fetching free agents: {str(e)}")
        
        # Convert to DataFrames and calculate PR values
        hitters_df = pd.DataFrame(hitter_stats)
        pitchers_df = pd.DataFrame(pitcher_stats)
        
        if not hitters_df.empty:
            hitters_df = self._calculate_pr_values(hitters_df, is_hitter=True)
            hitter_stats = hitters_df.to_dict('records')
        
        if not pitchers_df.empty:
            pitchers_df = self._calculate_pr_values(pitchers_df, is_hitter=False)
            pitcher_stats = pitchers_df.to_dict('records')
        
        logger.info(f"Found {len(hitter_stats)} hitters and {len(pitcher_stats)} pitchers")
        logger.info(f"Total unique players processed: {len(processed_players)}")
        return {
            'hitters': hitter_stats,
            'pitchers': pitcher_stats
        }

    def fetch_free_agents(self):
        """Fetch free agent stats from ESPN."""
        all_free_agents = []
        batch_size = 50  # ESPN API default size is 50
        offset = 0
        
        try:
            # Keep fetching until we get a batch with fewer players than the batch size
            while True:
                free_agents = self.league.free_agents(offset=offset)
                if not free_agents:
                    break
                
                all_free_agents.extend(free_agents)
                
                # If we got fewer players than the batch size, we've reached the end
                if len(free_agents) < batch_size:
                    break
                
                # Increment offset for next batch
                offset += batch_size
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            logging.info(f"Successfully fetched {len(all_free_agents)} free agents")
            return all_free_agents
        except Exception as e:
            logging.error(f"Error fetching free agents: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    league_id = int(os.getenv('LEAGUE_ID'))
    year = int(os.getenv('YEAR', 2025))  # Default to 2025
    
    fetcher = FantasyStatsFetcher(league_id, year)
    stats = fetcher.get_player_stats()
    
    # Convert to pandas DataFrames
    hitters_df = pd.DataFrame(stats['hitters'])
    pitchers_df = pd.DataFrame(stats['pitchers'])
    
    # Save to CSV files (for easy inspection)
    hitters_df.to_csv('data/hitter_stats.csv', index=False)
    pitchers_df.to_csv('data/pitcher_stats.csv', index=False)
    
    # Save to Postgres database
    fetcher._save_to_sqlite(stats)
    
    print(f"Saved stats for {len(hitters_df)} hitters and {len(pitchers_df)} pitchers")
    print(f"Data saved to both CSV files and Postgres database at {fetcher.db_url}") 