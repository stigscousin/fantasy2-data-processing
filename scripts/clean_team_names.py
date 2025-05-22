from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from pathlib import Path
import re
from sqlalchemy import create_engine
import os

def truncate_repeated_name(name):
    # Find the shortest substring that repeats
    for i in range(1, len(name)//2 + 1):
        candidate = name[:i]
        if candidate and name == candidate * (len(name) // len(candidate)):
            return candidate
        # If the name starts with candidate and the next occurrence is at i
        if name[i:].startswith(candidate):
            return candidate
    # Fallback: try to find the first repeated sequence
    match = re.match(r'(.+?)\1+', name)
    if match:
        return match.group(1)
    return name

def main():
    db_url = os.environ['DATABASE_URL']
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    engine = create_engine(db_url)
    df = pd.read_sql('SELECT rowid, team_name FROM team_stats', engine)
    
    cleaned = []
    for idx, row in df.iterrows():
        cleaned_name = truncate_repeated_name(row['team_name'])
        cleaned.append((cleaned_name, row['rowid']))
    
    # Update the table
    for name, rowid in cleaned:
        engine.execute('UPDATE team_stats SET team_name = %s WHERE rowid = %s', (name, rowid))
    print(f"Cleaned {len(cleaned)} team names.")

if __name__ == "__main__":
    main() 