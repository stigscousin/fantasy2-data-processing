import sqlite3
import pandas as pd
import logging
import psycopg2
from sqlalchemy import create_engine
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_team_stats_table(cursor):
    """Create the team_stats table with required columns."""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS team_stats (
        team_id INTEGER PRIMARY KEY,
        team_name TEXT NOT NULL,
        -- Current stats
        R INTEGER,
        HR INTEGER,
        TB INTEGER,
        SBN INTEGER,
        AVG REAL,
        OPS REAL,
        AB INTEGER,
        -- Projected stats
        proj_R INTEGER,
        proj_HR INTEGER,
        proj_TB INTEGER,
        proj_SBN INTEGER,
        proj_AVG REAL,
        proj_OPS REAL,
        proj_K INTEGER,
        proj_QS INTEGER,
        proj_ERA REAL,
        proj_WHIP REAL,
        proj_K_BB REAL,
        proj_SVHD INTEGER,
        proj_AB INTEGER,
        proj_IP REAL,
        -- Points
        R_points INTEGER,
        HR_points INTEGER,
        TB_points INTEGER,
        SBN_points INTEGER,
        AVG_points INTEGER,
        OPS_points INTEGER,
        Total_Points INTEGER DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    logger.info("Team stats table created successfully")

def create_team_stats_by_week_table(cursor):
    """Create the team_stats_by_week table with required columns."""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS team_stats_by_week (
        team_id INTEGER,
        team_name TEXT NOT NULL,
        week INTEGER NOT NULL,
        R INTEGER,
        HR INTEGER,
        TB INTEGER,
        SB INTEGER,
        CS INTEGER,
        H INTEGER,
        AB INTEGER,
        BB INTEGER,
        HBP INTEGER,
        SF INTEGER,
        K INTEGER,
        QS INTEGER,
        ER INTEGER,
        OUTS INTEGER,
        P_BB INTEGER,
        P_H INTEGER,
        SVHD INTEGER,
        OPS REAL,
        OPS_AB INTEGER,
        IP REAL,
        GS INTEGER,
        PRIMARY KEY (team_id, week)
    )
    ''')
    logger.info("Team stats by week table created successfully")

def init_database():
    """Initialize the Postgres database with proper schema."""
    try:
        # Connect to the database
        db_url = os.environ['DATABASE_URL']
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Drop hitters table if it exists to ensure schema is up to date
        cursor.execute('DROP TABLE IF EXISTS hitters')
        # Create hitters table with blended_AB and blended_G
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hitters (
            name TEXT,
            position TEXT,
            eligible_positions TEXT,
            team TEXT,
            player_id INTEGER,
            fangraphs_id TEXT,
            mlbamid TEXT,
            is_free_agent INTEGER,
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
            AVG_diff REAL,
            OPS_diff REAL,
            R_z REAL,
            HR_z REAL,
            TB_z REAL,
            SBN_z REAL,
            AVG_z REAL,
            OPS_z REAL,
            PR REAL,
            current_PR REAL,
            blended_AVG REAL,
            blended_OPS REAL,
            blended_R REAL,
            blended_HR REAL,
            blended_TB REAL,
            blended_SBN REAL,
            blended_PR REAL,
            blended_AVG_diff REAL,
            blended_OPS_diff REAL,
            blended_R_z REAL,
            blended_HR_z REAL,
            blended_TB_z REAL,
            blended_SBN_z REAL,
            blended_AVG_z REAL,
            blended_OPS_z REAL,
            blended_AB REAL,
            blended_G REAL,
            proj_G REAL,
            proj_PA REAL,
            proj_AB REAL,
            proj_H REAL,
            proj_2B REAL,
            proj_3B REAL,
            proj_HR REAL,
            proj_R REAL,
            proj_RBI REAL,
            proj_BB REAL,
            proj_SO REAL,
            proj_SB REAL,
            proj_CS REAL,
            proj_AVG REAL,
            proj_wOBA REAL,
            proj_OPS REAL,
            proj_wRC_plus REAL,
            proj_WAR REAL,
            proj_TB REAL,
            proj_SBN REAL,
            proj_AVG_diff REAL,
            proj_OPS_diff REAL,
            proj_R_z REAL,
            proj_HR_z REAL,
            proj_TB_z REAL,
            proj_SBN_z REAL,
            proj_AVG_z REAL,
            proj_OPS_z REAL,
            proj_PR REAL
        )
        ''')
        
        # Create hitter_impacts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hitter_impacts (
            player_id INTEGER,
            team_name TEXT,
            impact REAL,
            stat_type TEXT DEFAULT 'projected',
            PRIMARY KEY (player_id, team_name, stat_type)
        )
        ''')
        
        # Create pitcher_impacts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pitcher_impacts (
            player_id INTEGER,
            team_name TEXT,
            impact REAL,
            stat_type TEXT DEFAULT 'projected',
            PRIMARY KEY (player_id, team_name, stat_type)
        )
        ''')
        
        # Print the schema for verification (Postgres version)
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'hitters'")
        logger.info('Hitters table schema: %s', cursor.fetchall())
        
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
        
        # Create team stats table
        create_team_stats_table(cursor)
        
        # Create team stats by week table
        create_team_stats_by_week_table(cursor)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_database() 