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
    """Download FanGraphs projections using Selenium with stealth options"""
    print("Environment variables loaded:")
    fangraphs_username = os.getenv('FANGRAPHS_USERNAME')
    fangraphs_password = os.getenv('FANGRAPHS_PASSWORD')

    print(f"FANGRAPHS_USERNAME: {repr(fangraphs_username)} (type: {type(fangraphs_username)})")
    print(f"FANGRAPHS_PASSWORD: {'***' if fangraphs_password else None} (type: {type(fangraphs_password)})")

    # Create projections directory if it doesn't exist
    os.makedirs(PROJECTIONS_DIR, exist_ok=True)

    if not fangraphs_username or not fangraphs_password:
        print("Missing FanGraphs credentials!")
        raise Exception("No FanGraphs credentials available")

    try:
        # Set up Selenium for projections download with stealth options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            """
        })

        try:
            # 1. Go to login page
            print("Opening FanGraphs login page...")
            login_url = "https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/"
            driver.get(login_url)
            wait = WebDriverWait(driver, 30)

            # 2. Fill in login form
            print("Filling in login form...")
            user_input = wait.until(EC.presence_of_element_located((By.ID, "user_login")))
            pass_input = wait.until(EC.presence_of_element_located((By.ID, "user_pass")))
            submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "wp-submit")))
            user_input.clear()
            user_input.send_keys(fangraphs_username)
            pass_input.clear()
            pass_input.send_keys(fangraphs_password)
            submit_btn.click()

            # 3. Wait for redirect to homepage
            print("Waiting for login to complete...")
            wait.until(EC.url_contains("fangraphs.com"))
            print(f"Current URL after login: {driver.current_url}")

            # 4. Go to batters projections page
            print("Navigating to batters projections page...")
            batters_url = "https://www.fangraphs.com/projections?pos=all&stats=bat&type=ratcdc"
            print(f"Target URL: {batters_url}")
            print(f"Current URL before navigation: {driver.current_url}")
            
            driver.get(batters_url)
            print(f"Current URL immediately after navigation: {driver.current_url}")
            
            print("Waiting for batters page to load...")
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Debug: Print page info
            print(f"Current URL after wait: {driver.current_url}")
            print(f"Page title: {driver.title}")
            print(f"Page source length: {len(driver.page_source)}")
            print("Page source preview:")
            print(driver.page_source[:2000])
            
            # Check if we're still on the login page or homepage
            if "wp-login.php" in driver.current_url:
                print("ERROR: Still on login page - login may have failed")
                raise Exception("Still on login page after navigation")
            elif driver.current_url == "https://www.fangraphs.com/":
                print("ERROR: Redirected to homepage - may need additional authentication")
                raise Exception("Redirected to homepage instead of projections page")
            elif "projections" not in driver.current_url:
                print(f"WARNING: Not on projections page. Current URL: {driver.current_url}")
            
            # Wait a bit more for any dynamic content to load
            print("Waiting additional time for page content to load...")
            time.sleep(5)
            print(f"Final URL after additional wait: {driver.current_url}")
            print(f"Final page title: {driver.title}")

            # 5. Find and click export button with multiple selectors
            print("Looking for export button...")
            export_button = None
            
            # Try different selectors
            selectors = [
                (By.XPATH, "//button[contains(text(), 'Export Data')]"),
                (By.XPATH, "//button[contains(text(), 'Export')]"),
                (By.XPATH, "//button[contains(@class, 'export')]"),
                (By.CSS_SELECTOR, "button[data-testid='export-button']"),
                (By.CSS_SELECTOR, "button.export"),
                (By.XPATH, "//*[contains(text(), 'Export Data')]"),
                (By.XPATH, "//*[contains(text(), 'Export')]")
            ]
            
            for selector_type, selector in selectors:
                try:
                    print(f"Trying selector: {selector_type} = {selector}")
                    export_button = wait.until(EC.element_to_be_clickable((selector_type, selector)))
                    print(f"Found export button with selector: {selector}")
                    break
                except Exception as e:
                    print(f"Selector failed: {e}")
                    continue
            
            if export_button:
                export_button.click()
                print("Clicked export button for batters.")
            else:
                print("Could not find export button with any selector")
                raise Exception("Export button not found")

            # 6. Move downloaded file
            downloads_dir = Path.home() / "Downloads"
            time.sleep(3)  # Wait for download to complete
            
            # Find the most recent fangraphs CSV file
            fangraphs_files = list(downloads_dir.glob("fangraphs-leaderboard-projections*.csv"))
            if fangraphs_files:
                # Sort by modification time and get the most recent
                fangraphs_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                batters_file = fangraphs_files[0]
                batters_path = os.path.join(PROJECTIONS_DIR, 'fangraphs-leaderboard-projections-batters.csv')
                batters_file.rename(batters_path)
                print(f"Batters projections saved: {batters_path}")
            else:
                print("Batters projections file not found in downloads")

            # 7. Repeat for pitchers
            print("Navigating to pitchers projections page...")
            pitchers_url = "https://www.fangraphs.com/projections?type=ratcdc&stats=pit"
            print(f"Target URL: {pitchers_url}")
            print(f"Current URL before navigation: {driver.current_url}")
            
            driver.get(pitchers_url)
            print(f"Current URL immediately after navigation: {driver.current_url}")
            
            print("Waiting for pitchers page to load...")
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Debug: Print page info
            print(f"Current URL after wait: {driver.current_url}")
            print(f"Page title: {driver.title}")
            print(f"Page source length: {len(driver.page_source)}")
            
            # Check if we're still on the login page or homepage
            if "wp-login.php" in driver.current_url:
                print("ERROR: Still on login page - login may have failed")
                raise Exception("Still on login page after navigation")
            elif driver.current_url == "https://www.fangraphs.com/":
                print("ERROR: Redirected to homepage - may need additional authentication")
                raise Exception("Redirected to homepage instead of projections page")
            elif "projections" not in driver.current_url:
                print(f"WARNING: Not on projections page. Current URL: {driver.current_url}")
            
            # Wait a bit more for any dynamic content to load
            print("Waiting additional time for page content to load...")
            time.sleep(5)
            print(f"Final URL after additional wait: {driver.current_url}")
            print(f"Final page title: {driver.title}")

            # 8. Find and click export button with multiple selectors
            print("Looking for export button...")
            export_button = None
            
            # Try different selectors
            selectors = [
                (By.XPATH, "//button[contains(text(), 'Export Data')]"),
                (By.XPATH, "//button[contains(text(), 'Export')]"),
                (By.XPATH, "//button[contains(@class, 'export')]"),
                (By.CSS_SELECTOR, "button[data-testid='export-button']"),
                (By.CSS_SELECTOR, "button.export"),
                (By.XPATH, "//*[contains(text(), 'Export Data')]"),
                (By.XPATH, "//*[contains(text(), 'Export')]")
            ]
            
            for selector_type, selector in selectors:
                try:
                    print(f"Trying selector: {selector_type} = {selector}")
                    export_button = wait.until(EC.element_to_be_clickable((selector_type, selector)))
                    print(f"Found export button with selector: {selector}")
                    break
                except Exception as e:
                    print(f"Selector failed: {e}")
                    continue
            
            if export_button:
                export_button.click()
                print("Clicked export button for pitchers.")
            else:
                print("Could not find export button with any selector")
                raise Exception("Export button not found")
            
            time.sleep(5)

            # Find the most recent fangraphs CSV file (should be the pitchers file now)
            fangraphs_files = list(downloads_dir.glob("fangraphs-leaderboard-projections*.csv"))
            if fangraphs_files:
                # Sort by modification time and get the most recent
                fangraphs_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                pitchers_file = fangraphs_files[0]
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