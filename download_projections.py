import os
import time
import pickle
from pathlib import Path
from datetime import datetime
from curl_cffi import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
COOKIES_PATH = 'fangraphs_cookies.pkl'
PROJECTIONS_DIR = 'projections'

def save_cookies(session, path):
    """Save cookies from session to file"""
    try:
        # Convert curl_cffi cookies to a format we can pickle
        cookies_to_save = []
        for cookie in session.cookies:
            cookies_to_save.append({
                'name': cookie.name,
                'value': cookie.value,
                'domain': cookie.domain,
                'path': cookie.path
            })
        
        with open(path, 'wb') as f:
            pickle.dump(cookies_to_save, f)
        print(f"Cookies saved to {path}")
    except Exception as e:
        print(f"Error saving cookies: {e}")

def load_cookies(session, path):
    """Load cookies from file into session"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                cookies = pickle.load(f)
            
            # Convert cookies to curl_cffi format
            for cookie in cookies:
                session.cookies.set(
                    name=cookie['name'],
                    value=cookie['value'],
                    domain=cookie['domain'],
                    path=cookie['path']
                )
            print(f"Cookies loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading cookies: {e}")
            return False
    return False

def verify_login(session):
    """Verify if we're logged in by checking for member indicators"""
    try:
        # Try to access a page that requires login
        response = session.get("https://www.fangraphs.com/", timeout=10)
        if response.status_code == 200:
            # Check for login indicators in the response
            if "fg_is_member" in response.text or "logout" in response.text.lower():
                print("Login verification successful")
                return True
            else:
                print("Login verification failed - no member indicators found")
                return False
        else:
            print(f"Login verification failed - status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Login verification error: {e}")
        return False

def download_projections():
    """Download FanGraphs projections using curl_cffi"""
    print("Environment variables loaded:")
    print(f"FANGRAPHS_USERNAME: \"{os.getenv('FANGRAPHS_USERNAME')}\"")
    print(f"FANGRAPHS_PASSWORD: \"***\"")
    
    # Create projections directory if it doesn't exist
    os.makedirs(PROJECTIONS_DIR, exist_ok=True)
    
    # Set up session with curl_cffi
    session = requests.Session()
    
    # Configure session for better anti-bot bypass
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Check if we have FanGraphs credentials
    fangraphs_username = os.getenv('FANGRAPHS_USERNAME')
    fangraphs_password = os.getenv('FANGRAPHS_PASSWORD')
    
    if not fangraphs_username or not fangraphs_password:
        raise Exception("No FanGraphs credentials available")
    
    try:
        # Try to load existing cookies first
        if load_cookies(session, COOKIES_PATH):
            if verify_login(session):
                print("Using existing login session")
            else:
                print("Existing cookies expired, attempting fresh login...")
                # Clear cookies and try fresh login
                session.cookies.clear()
        else:
            print("No existing cookies found, attempting fresh login...")
        
        # Perform fresh login if needed
        if not verify_login(session):
            print("Attempting fresh login with curl_cffi...")
            
            # First, get the login page to extract any necessary tokens
            login_url = "https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/"
            response = session.get(login_url, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to load login page: {response.status_code}")
                raise Exception("Could not load login page")
            
            print(f"Login page loaded successfully (status: {response.status_code})")
            
            # Extract nonce if present (WordPress security token)
            nonce = None
            if 'name="_wpnonce"' in response.text:
                import re
                nonce_match = re.search(r'name="_wpnonce" value="([^"]+)"', response.text)
                if nonce_match:
                    nonce = nonce_match.group(1)
                    print(f"Found WordPress nonce: {nonce}")
            
            # Prepare login data
            login_data = {
                'log': fangraphs_username,
                'pwd': fangraphs_password,
                'wp-submit': 'Log In',
                'redirect_to': 'https://www.fangraphs.com/',
                'testcookie': '1'
            }
            
            if nonce:
                login_data['_wpnonce'] = nonce
            
            # Submit login form
            print("Submitting login form...")
            login_response = session.post(login_url, data=login_data, timeout=10)
            
            print(f"Login response status: {login_response.status_code}")
            print(f"Login response URL: {login_response.url}")
            
            # Check if login was successful
            if verify_login(session):
                print("Login successful!")
                save_cookies(session, COOKIES_PATH)
            else:
                print("Login failed - verification unsuccessful")
                print(f"Response content preview: {login_response.text[:500]}")
                raise Exception("Login failed")
        
        # Now download the projections
        print("Login successful, downloading projections...")
        
        # Download batters projections
        batters_url = "https://www.fangraphs.com/api/projections"
        batters_params = {
            'type': '0',
            'pos': 'all',
            'stats': 'bat',
            'qual': '0',
            'sort': '31,d',
            'season': '2025',
            'team': '0,ts',
            'rost': '0',
            'filter': '',
            'players': '0',
            'pg': '0'
        }
        
        print("Downloading batters projections...")
        batters_response = session.get(batters_url, params=batters_params, timeout=30)
        
        if batters_response.status_code == 200:
            batters_df = pd.read_csv(batters_response.content)
            batters_path = os.path.join(PROJECTIONS_DIR, 'fangraphs-leaderboard-projections-batters.csv')
            batters_df.to_csv(batters_path, index=False)
            print(f"Batters projections saved: {batters_path} ({len(batters_df)} players)")
        else:
            print(f"Failed to download batters projections: {batters_response.status_code}")
            print(f"Response: {batters_response.text[:500]}")
            raise Exception("Failed to download batters projections")
        
        # Download pitchers projections
        pitchers_params = {
            'type': '0',
            'pos': 'all',
            'stats': 'pit',
            'qual': '0',
            'sort': '31,d',
            'season': '2025',
            'team': '0,ts',
            'rost': '0',
            'filter': '',
            'players': '0',
            'pg': '0'
        }
        
        print("Downloading pitchers projections...")
        pitchers_response = session.get(batters_url, params=pitchers_params, timeout=30)
        
        if pitchers_response.status_code == 200:
            pitchers_df = pd.read_csv(pitchers_response.content)
            pitchers_path = os.path.join(PROJECTIONS_DIR, 'fangraphs-leaderboard-projections-pitchers.csv')
            pitchers_df.to_csv(pitchers_path, index=False)
            print(f"Pitchers projections saved: {pitchers_path} ({len(pitchers_df)} players)")
        else:
            print(f"Failed to download pitchers projections: {pitchers_response.status_code}")
            print(f"Response: {pitchers_response.text[:500]}")
            raise Exception("Failed to download pitchers projections")
        
        print("All projections downloaded successfully!")
        
    except Exception as e:
        print(f"Error during process: {e}")
        raise

if __name__ == "__main__":
    download_projections() 