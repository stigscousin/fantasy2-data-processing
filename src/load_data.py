import pandas as pd
import logging
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def load_data():
    """Load data from CSV files into the Postgres database."""
    try:
        # Connect to the database
        db_url = os.environ['DATABASE_URL']
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        engine = create_engine(db_url)
        
        # Read CSV files
        hitters_df = pd.read_csv('data/hitter_stats.csv')
        pitchers_df = pd.read_csv('data/pitcher_stats.csv')
        
        # Load data into tables
        hitters_df.to_sql('hitters', engine, if_exists='replace', index=False)
        pitchers_df.to_sql('pitchers', engine, if_exists='replace', index=False)
        
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    load_data() 