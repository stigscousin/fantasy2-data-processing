from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import os
import shutil
from datetime import datetime
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from dotenv import load_dotenv
import tempfile
import pickle

# Load environment variables
load_dotenv()

# Debug: Check if environment variables are loaded
print("Environment variables loaded:")
print("FANGRAPHS_USERNAME:", os.getenv('FANGRAPHS_USERNAME'))
print("FANGRAPHS_PASSWORD:", "***" if os.getenv('FANGRAPHS_PASSWORD') else "Not set")

COOKIES_PATH = "fangraphs_cookies.pkl"

def save_cookies(driver, path):
    with open(path, 'wb') as filehandler:
        pickle.dump(driver.get_cookies(), filehandler)

def load_cookies(driver, path):
    with open(path, 'rb') as cookiesfile:
        cookies = pickle.load(cookiesfile)
        for cookie in cookies:
            # Selenium requires expiry to be int, not float
            if isinstance(cookie.get('expiry', None), float):
                cookie['expiry'] = int(cookie['expiry'])
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                print(f"Could not add cookie: {cookie}, error: {e}")

def find_export_button(driver):
    """Try multiple strategies to find the export button"""
    strategies = [
        (By.CSS_SELECTOR, "a.data-export"),  # Class-based - most likely match
        (By.CSS_SELECTOR, "a[href*='data:application/csv']"),  # href-based
        (By.CSS_SELECTOR, "a[href*='Export']"),  # Text in href
        (By.XPATH, "//a[contains(@class, 'data-export')]"),  # XPath class
        (By.XPATH, "//a[contains(text(), 'Export')]"),  # Text content
    ]
    
    for by, selector in strategies:
        try:
            print(f"Trying to find export button with {by}: {selector}")
            # First check if element exists
            elements = driver.find_elements(by, selector)
            if elements:
                print(f"Found {len(elements)} potential export buttons")
                for element in elements:
                    try:
                        print(f"Button text: {element.text}")
                        print(f"Button href: {element.get_attribute('href')}")
                        print(f"Button class: {element.get_attribute('class')}")
                        if element.is_displayed() and element.is_enabled():
                            print(f"Found clickable button using {by}: {selector}")
                            return element
                    except:
                        continue
            else:
                print(f"No elements found with {by}: {selector}")
        except Exception as e:
            print(f"Error trying {by}: {selector} - {str(e)}")
            continue
    
    return None

def verify_login(driver):
    """Verify that we're actually logged in"""
    try:
        # Try to find a logged-in element
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="logout"]'))
        )
        print("Successfully verified login")
        return True
    except:
        print("Could not verify login - might not be logged in")
        return False

def download_projections_for_type(driver, player_type="batters"):
    """Download projections for either batters or pitchers"""
    print(f"Downloading {player_type} projections...")
    
    if player_type == "pitchers":
        print("Navigating to pitchers page...")
        try:
            # Navigate directly to the pitchers page
            driver.get('https://www.fangraphs.com/projections?type=ratcdc&stats=pit&pos=all&team=0&players=0&lg=all&z=1744973723&pageitems=30&statgroup=dashboard&fantasypreset=dashboard')
            time.sleep(5)  # Wait for page to load
            
            # Verify we're on the pitchers page
            if "stats=pit" not in driver.current_url:
                raise Exception("Failed to navigate to pitchers page")
            
        except Exception as e:
            print(f"Error navigating to pitchers page: {str(e)}")
            driver.save_screenshot(f'error_pitchers_page.png')
            print("Current URL:", driver.current_url)
            print("Page source:")
            print(driver.page_source)
            raise
    
    print("Looking for table...")
    # Wait for the table to be present
    table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="table-wrapper"]'))
    )
    
    print("Table found. Looking for export button...")
    export_button = find_export_button(driver)
    
    if not export_button:
        print("Could not find export button. Taking screenshot...")
        driver.save_screenshot(f'projections_page_{player_type}.png')
        print("\nPage source:")
        print(driver.page_source)
        print("\nAll elements with 'a' tag:")
        links = driver.find_elements(By.TAG_NAME, 'a')
        for link in links:
            print(f"Link text: {link.text}, href: {link.get_attribute('href')}")
        raise Exception(f"Could not find export button for {player_type}")
    
    print("Clicking export button...")
    # Try to click using JavaScript
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
        time.sleep(2)
        driver.execute_script("arguments[0].click();", export_button)
    except Exception as e:
        print(f"JavaScript click failed: {str(e)}")
        # Fallback to regular click
        export_button.click()
    
    print("Waiting for download...")
    time.sleep(5)
    
    # Rename the downloaded file to indicate player type
    # Check for both the original name and Chrome's auto-numbered version
    potential_old_files = [
        os.path.join(os.getcwd(), 'projections', 'fangraphs-leaderboard-projections.csv'),
        os.path.join(os.getcwd(), 'projections', 'fangraphs-leaderboard-projections (1).csv')
    ]
    new_file = os.path.join(os.getcwd(), 'projections', f'fangraphs-leaderboard-projections-{player_type}.csv')
    
    for old_file in potential_old_files:
        if os.path.exists(old_file):
            shutil.move(old_file, new_file)
            print(f"Renamed file to {new_file}")
            break
    
    print(f"Successfully downloaded {player_type} projections")

def download_projections():
    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    # Spoof user-agent to look like a real browser
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # Bright Data proxy configuration
    brightdata_username = os.getenv('BRIGHTDATA_USERNAME')
    brightdata_password = os.getenv('BRIGHTDATA_PASSWORD')
    brightdata_host = os.getenv('BRIGHTDATA_HOST', 'brd.superproxy.io')
    brightdata_port = os.getenv('BRIGHTDATA_PORT', '22225')
    
    if brightdata_username and brightdata_password:
        print("Using Bright Data proxy...")
        proxy_string = f"http://{brightdata_username}-country-us:{brightdata_password}@{brightdata_host}:{brightdata_port}"
        chrome_options.add_argument(f'--proxy-server={proxy_string}')
    else:
        print("Warning: Bright Data credentials not found. Running without proxy.")

    # Set download directory
    download_dir = os.path.join(os.getcwd(), 'projections')
    os.makedirs(download_dir, exist_ok=True)
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "download.default_directory": download_dir,
        # Additional preferences to handle data URLs
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": False,
        "browser.helperApps.neverAsk.saveToDisk": "application/csv,text/csv"
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get("https://www.fangraphs.com/")
        time.sleep(3)
        
        # Check if we have FanGraphs credentials in environment variables
        fangraphs_username = os.getenv('FANGRAPHS_USERNAME')
        fangraphs_password = os.getenv('FANGRAPHS_PASSWORD')
        
        # Always attempt fresh login - skip cookie loading
        if fangraphs_username and fangraphs_password:
            print("Attempting fresh login with environment credentials...")
            try:
                # Navigate to the WordPress login page
                driver.get("https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/")
                time.sleep(3)
                
                # Debug: Print current URL and page title
                print(f"Current URL: {driver.current_url}")
                print(f"Page title: {driver.title}")
                
                # Debug: Check if we can find any form elements
                try:
                    forms = driver.find_elements(By.TAG_NAME, "form")
                    print(f"Found {len(forms)} form(s) on the page")
                    for i, form in enumerate(forms):
                        print(f"Form {i}: action='{form.get_attribute('action')}', method='{form.get_attribute('method')}'")
                except Exception as e:
                    print(f"Error finding forms: {e}")
                
                # Debug: Print page source to see what's actually loaded
                try:
                    page_source = driver.page_source
                    print(f"Page source length: {len(page_source)} characters")
                    print("First 500 characters of page source:")
                    print(page_source[:500])
                    
                    # Check for common indicators
                    if "cloudflare" in page_source.lower():
                        print("⚠️  Cloudflare detected in page source!")
                    if "captcha" in page_source.lower():
                        print("⚠️  Captcha detected in page source!")
                    if "blocked" in page_source.lower():
                        print("⚠️  Blocked page detected!")
                    if "security" in page_source.lower():
                        print("⚠️  Security check detected!")
                except Exception as e:
                    print(f"Error getting page source: {e}")
                
                # Try to find and fill login form
                username_field = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "log"))
                )
                password_field = driver.find_element(By.NAME, "pwd")
                
                username_field.send_keys(fangraphs_username)
                password_field.send_keys(fangraphs_password)
                
                # Submit the form
                submit_button = driver.find_element(By.ID, "wp-submit")
                submit_button.click()
                
                time.sleep(5)
                
                # Verify login was successful
                if verify_login(driver):
                    print("Automated login successful!")
                    save_cookies(driver, COOKIES_PATH)
                else:
                    print("Automated login failed - login verification failed")
                    raise Exception("Login verification failed")
                    
            except Exception as e:
                print(f"Automated login failed: {e}")
                raise Exception("Automated login failed")
        else:
            print("No FanGraphs credentials found in environment variables")
            raise Exception("No FanGraphs credentials available")
        
        # Now continue as if logged in
        print("Navigating to projections page...")
        driver.get('https://www.fangraphs.com/projections?pos=all&stats=bat&type=ratcdc')
        print("Waiting for page to load...")
        time.sleep(5)
        print("Current URL:", driver.current_url)
        download_projections_for_type(driver, "batters")
        print("Navigating to pitchers page...")
        driver.get('https://www.fangraphs.com/projections?type=ratcdc&stats=pit&pos=all&team=0&players=0&lg=all&z=1744973723&pageitems=30&statgroup=dashboard&fantasypreset=dashboard')
        time.sleep(5)
        if "stats=pit" not in driver.current_url:
            raise Exception("Failed to navigate to pitchers page")
        print("Looking for table...")
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="table-wrapper"]'))
        )
        print("Table found. Looking for export button...")
        export_button = find_export_button(driver)
        if not export_button:
            print("Could not find export button. Taking screenshot...")
            driver.save_screenshot(f'projections_page_pitchers.png')
            print("\nPage source:")
            print(driver.page_source)
            print("\nAll elements with 'a' tag:")
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                print(f"Link text: {link.text}, href: {link.get_attribute('href')}")
            raise Exception("Could not find export button for pitchers")
        print("Clicking export button...")
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
            time.sleep(2)
            driver.execute_script("arguments[0].click();", export_button)
        except Exception as e:
            print(f"JavaScript click failed: {str(e)}")
            export_button.click()
        print("Waiting for download...")
        time.sleep(5)
        potential_old_files = [
            os.path.join(os.getcwd(), 'projections', 'fangraphs-leaderboard-projections.csv'),
            os.path.join(os.getcwd(), 'projections', 'fangraphs-leaderboard-projections (1).csv')
        ]
        new_file = os.path.join(os.getcwd(), 'projections', 'fangraphs-leaderboard-projections-pitchers.csv')
        for old_file in potential_old_files:
            if os.path.exists(old_file):
                shutil.move(old_file, new_file)
                print(f"Renamed file to {new_file}")
                break
        print("Successfully downloaded pitchers projections")
    except Exception as e:
        print(f"Error during process: {str(e)}")
        print("Current URL:", driver.current_url)
        driver.save_screenshot('error_state.png')
        raise
    finally:
        driver.quit()

if __name__ == "__main__":
    download_projections() 