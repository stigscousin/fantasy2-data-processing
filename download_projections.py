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
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Load environment variables
load_dotenv()

# Constants
COOKIES_PATH = 'fangraphs_cookies.pkl'
PROJECTIONS_DIR = 'projections'

def save_cookies(session, path):
    """Save session cookies to file"""
    try:
        cookies = []
        for cookie in session.cookies:
            cookies.append({
                'name': cookie.name,
                'value': cookie.value,
                'domain': cookie.domain,
                'path': cookie.path,
                'secure': cookie.secure,
                'expires': cookie.expires
            })
        with open(path, 'wb') as f:
            pickle.dump(cookies, f)
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
        
        # Set up Selenium for projections download
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Add proxy if configured
        if proxy_url:
            chrome_options.add_argument(f'--proxy-server={proxy_url}')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Transfer cookies from curl_cffi session to Selenium
            print("Transferring cookies to Selenium...")
            driver.get("https://www.fangraphs.com")
            for cookie in session.cookies:
                try:
                    # Handle curl_cffi cookie objects which might be strings or have different attributes
                    if hasattr(cookie, 'name') and hasattr(cookie, 'value'):
                        driver.add_cookie({
                            'name': cookie.name,
                            'value': cookie.value,
                            'domain': getattr(cookie, 'domain', ''),
                            'path': getattr(cookie, 'path', '/'),
                            'secure': getattr(cookie, 'secure', False)
                        })
                    else:
                        print(f"Skipping cookie with unexpected format: {cookie}")
                except Exception as e:
                    print(f"Error adding cookie: {e}")
            
            # Download batters projections
            print("Downloading batters projections...")
            batters_url = "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&type=8&season=2025&month=0&season1=2025&ind=0&team=0,ts&rost=0&age=0&filter=&players=0&startdate=&enddate=&page=1_50"
            driver.get(batters_url)
            
            # Wait for page to load and take screenshot for debugging
            print("Waiting for page to load...")
            time.sleep(10)
            
            # Take screenshot for debugging
            driver.save_screenshot("/tmp/fangraphs_page.png")
            print("Screenshot saved to /tmp/fangraphs_page.png")
            
            # Print page title and URL for debugging
            print(f"Page title: {driver.title}")
            print(f"Current URL: {driver.current_url}")
            
            # Try different selectors for the export button
            wait = WebDriverWait(driver, 30)
            export_button = None
            
            # Try multiple selectors
            selectors = [
                "//button[contains(text(), 'Export Data')]",
                "//button[contains(text(), 'Export')]",
                "//a[contains(text(), 'Export Data')]",
                "//a[contains(text(), 'Export')]",
                "//button[@class='export-button']",
                "//a[@class='export-button']",
                "//button[contains(@class, 'export')]",
                "//a[contains(@class, 'export')]"
            ]
            
            for selector in selectors:
                try:
                    print(f"Trying selector: {selector}")
                    export_button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                    print(f"Found export button with selector: {selector}")
                    break
                except Exception as e:
                    print(f"Selector {selector} failed: {e}")
                    continue
            
            if export_button:
                export_button.click()
                print("Clicked export button")
            else:
                print("Could not find export button with any selector")
                # Print page source for debugging
                print("Page source preview:")
                print(driver.page_source[:2000])
                raise Exception("Export button not found")
            
            # Wait for download to complete
            time.sleep(5)
            
            # Check if file was downloaded
            downloads_dir = Path.home() / "Downloads"
            batters_file = None
            for file in downloads_dir.glob("*.csv"):
                if "fangraphs" in file.name.lower() and "bat" in file.name.lower():
                    batters_file = file
                    break
            
            if batters_file:
                # Move to projections directory
                batters_path = os.path.join(PROJECTIONS_DIR, 'fangraphs-leaderboard-projections-batters.csv')
                batters_file.rename(batters_path)
                print(f"Batters projections saved: {batters_path}")
            else:
                print("Batters projections file not found in downloads")
            
            # Download pitchers projections
            print("Downloading pitchers projections...")
            pitchers_url = "https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all&qual=0&type=8&season=2025&month=0&season1=2025&ind=0&team=0,ts&rost=0&age=0&filter=&players=0&startdate=&enddate=&page=1_50"
            driver.get(pitchers_url)
            
            # Wait for page to load and find export button
            export_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Export Data')]")))
            export_button.click()
            
            # Wait for download to complete
            time.sleep(5)
            
            # Check if file was downloaded
            pitchers_file = None
            for file in downloads_dir.glob("*.csv"):
                if "fangraphs" in file.name.lower() and "pit" in file.name.lower():
                    pitchers_file = file
                    break
            
            if pitchers_file:
                # Move to projections directory
                pitchers_path = os.path.join(PROJECTIONS_DIR, 'fangraphs-leaderboard-projections-pitchers.csv')
                pitchers_file.rename(pitchers_path)
                print(f"Pitchers projections saved: {pitchers_path}")
            else:
                print("Pitchers projections file not found in downloads")
            
            print("All projections downloaded successfully!")
            
        finally:
            driver.quit()
        
    except Exception as e:
        print(f"Error during process: {e}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    download_projections() 