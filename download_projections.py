import os
import time
import pickle
from pathlib import Path
from datetime import datetime
from curl_cffi import requests
import pandas as pd
from dotenv import load_dotenv
import traceback
import urllib3

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
        # Try to access the login page to see if we're already logged in
        login_url = "https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/"
        response = session.get(login_url, timeout=10)
        
        if response.status_code == 200:
            # If we're logged in, we should be redirected away from the login page
            # Check if we're still on the login page (which means we're not logged in)
            if "Log In" in response.text and "Username or Email Address" in response.text:
                print("Login verification failed - still on login page")
                return False
            elif "FanGraphs Baseball" in response.text and "Baseball Statistics and Analysis" in response.text:
                print("Login verification successful - redirected to FanGraphs homepage")
                return True
            else:
                print("Login verification unclear - checking response content")
                print(f"Response contains 'Log In': {'Log In' in response.text}")
                print(f"Response contains 'FanGraphs Baseball': {'FanGraphs Baseball' in response.text}")
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
    fangraphs_username = os.getenv('FANGRAPHS_USERNAME')
    fangraphs_password = os.getenv('FANGRAPHS_PASSWORD')
    brightdata_username = os.getenv('BRIGHTDATA_USERNAME')
    brightdata_password = os.getenv('BRIGHTDATA_PASSWORD')
    
    # Construct proxy URL from BrightData credentials
    proxy_url = None
    if brightdata_username and brightdata_password:
        # Try datacenter proxies first (no KYC required)
        proxy_url = f"http://{brightdata_username}:{brightdata_password}@brd.superproxy.io:22225"
        print(f"Constructed proxy URL from BrightData credentials (datacenter)")
    else:
        print("No BrightData credentials found.")
    
    print(f"FANGRAPHS_USERNAME: {repr(fangraphs_username)} (type: {type(fangraphs_username)})")
    print(f"FANGRAPHS_PASSWORD: {'***' if fangraphs_password else None} (type: {type(fangraphs_password)})")
    print(f"BRIGHTDATA_USERNAME: {repr(brightdata_username)} (type: {type(brightdata_username)})")
    print(f"BRIGHTDATA_PASSWORD: {'***' if brightdata_password else None} (type: {type(brightdata_password)})")
    print(f"PROXY_URL: {repr(proxy_url)} (type: {type(proxy_url)})")
    print(f"Other env: {[(k, v) for k, v in os.environ.items() if 'FANGRAPHS' in k or 'BRIGHTDATA' in k]}")

    # Create projections directory if it doesn't exist
    os.makedirs(PROJECTIONS_DIR, exist_ok=True)

    # Set up session with curl_cffi
    session = requests.Session()
    use_proxy = False
    if proxy_url:
        print(f"Using proxy: {proxy_url}")
        session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        # Configure SSL settings for proxy
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        use_proxy = True
    else:
        print("No proxy configured.")

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

    if not fangraphs_username or not fangraphs_password:
        print("Missing FanGraphs credentials!")
        raise Exception("No FanGraphs credentials available")

    try:
        # Try to load existing cookies first
        print("Loading cookies...")
        if load_cookies(session, COOKIES_PATH):
            print("Cookies loaded. Verifying login...")
            if verify_login(session):
                print("Using existing login session")
            else:
                print("Existing cookies expired, attempting fresh login...")
                session.cookies.clear()
        else:
            print("No existing cookies found, attempting fresh login...")

        # Perform fresh login if needed
        if not verify_login(session):
            print("Attempting fresh login with curl_cffi...")
            login_url = "https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/"
            print(f"GET {login_url}")
            response = session.get(login_url, timeout=10)
            print(f"Login page GET status: {response.status_code}")
            print(f"Login page headers: {dict(response.headers)}")
            print(f"Login page cookies: {dict(response.cookies)}")
            print(f"Cloudflare headers: {{k: v for k, v in response.headers.items() if 'cf-' in k.lower() or 'cloudflare' in k.lower()}}")
            print(f"Login page body (first 1000 chars): {response.text[:1000]}")

            # If proxy failed with 402, try without proxy
            if response.status_code == 402 and use_proxy:
                print("Proxy failed with 402 error, retrying without proxy...")
                session.proxies = {}
                session.verify = True
                response = session.get(login_url, timeout=10)
                print(f"Retry without proxy - status: {response.status_code}")
                print(f"Retry without proxy - headers: {dict(response.headers)}")

            if response.status_code != 200:
                print(f"Failed to load login page: {response.status_code}")
                raise Exception("Could not load login page")

            nonce = None
            if 'name="_wpnonce"' in response.text:
                import re
                nonce_match = re.search(r'name="_wpnonce" value="([^"]+)"', response.text)
                if nonce_match:
                    nonce = nonce_match.group(1)
                    print(f"Found WordPress nonce: {nonce}")

            login_data = {
                'log': fangraphs_username,
                'pwd': fangraphs_password,
                'wp-submit': 'Log In',
                'redirect_to': 'https://www.fangraphs.com/',
                'testcookie': '1'
            }
            if nonce:
                login_data['_wpnonce'] = nonce

            print(f"POST {login_url} with data: {{'log': fangraphs_username, 'pwd': '***', ...}}")
            login_response = session.post(login_url, data=login_data, timeout=10)
            print(f"Login POST status: {login_response.status_code}")
            print(f"Login POST headers: {dict(login_response.headers)}")
            print(f"Login POST cookies: {dict(login_response.cookies)}")
            print(f"Cloudflare headers: {{k: v for k, v in login_response.headers.items() if 'cf-' in k.lower() or 'cloudflare' in k.lower()}}")
            print(f"Login POST body (first 1000 chars): {login_response.text[:1000]}")

            # Check if login was successful
            if verify_login(session):
                print("Login successful!")
                save_cookies(session, COOKIES_PATH)
            else:
                # Check if we got redirected to the homepage (which indicates successful login)
                if "FanGraphs Baseball" in login_response.text and "Baseball Statistics and Analysis" in login_response.text:
                    print("Login successful! (detected homepage redirect)")
                    save_cookies(session, COOKIES_PATH)
                else:
                    print("Login failed - verification unsuccessful")
                    print(f"Response content preview: {login_response.text[:1000]}")
                    raise Exception("Login failed")

        print("Login successful, downloading projections...")
        
        # Now download the projections
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
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    download_projections() 