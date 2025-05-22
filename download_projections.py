from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os
import shutil
from datetime import datetime
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Debug: Check if environment variables are loaded
print("Environment variables loaded:")
print("FANGRAPHS_USERNAME:", os.getenv('FANGRAPHS_USERNAME'))
print("FANGRAPHS_PASSWORD:", "***" if os.getenv('FANGRAPHS_PASSWORD') else "Not set")

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
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    # Use a unique temp directory for user-data-dir
    user_data_dir = tempfile.mkdtemp()
    chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
    
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
        # Navigate to login page
        print("Navigating to login page...")
        driver.get("https://blogs.fangraphs.com/wp-login.php?redirect_to=https://www.fangraphs.com/")
        
        # Wait for page to load
        time.sleep(5)
        
        print("Current URL:", driver.current_url)
        print("Looking for login form...")
        
        try:
            # Try multiple selectors for the username field
            selectors = [
                (By.ID, "user_login"),
                (By.NAME, "log"),
                (By.CSS_SELECTOR, "input#user_login")
            ]
            
            username_field = None
            for by, selector in selectors:
                try:
                    print(f"Trying selector: {by} = {selector}")
                    username_field = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    if username_field:
                        print(f"Found username field with {by} = {selector}")
                        break
                except:
                    continue
            
            if not username_field:
                print("Could not find username field. Taking screenshot...")
                driver.save_screenshot('login_form_not_found.png')
                print("Page source:")
                print(driver.page_source)
                raise Exception("Could not find username field")
            
            # Now find password field and submit button
            password_field = driver.find_element(By.ID, "user_pass")
            submit_button = driver.find_element(By.ID, "wp-submit")
            
            username = os.getenv('FANGRAPHS_USERNAME')
            password = os.getenv('FANGRAPHS_PASSWORD')
            
            if not username or not password:
                raise ValueError("FANGRAPHS_USERNAME and FANGRAPHS_PASSWORD environment variables must be set")
            
            print("Found all form elements, filling in credentials...")
            
            username_field.clear()
            username_field.send_keys(username)
            
            password_field.clear()
            password_field.send_keys(password)
            
            print("Submitting form...")
            submit_button.click()
            
            print("Waiting for login to complete...")
            time.sleep(5)
            
        except Exception as e:
            print(f"Error during login: {str(e)}")
            print("Current URL:", driver.current_url)
            driver.save_screenshot('login_error.png')
            raise
        
        # Verify login was successful
        if not verify_login(driver):
            print("Login verification failed. Taking screenshot...")
            driver.save_screenshot('login_failed.png')
            print("Current URL:", driver.current_url)
            print("Page source:")
            print(driver.page_source)
            raise Exception("Login verification failed")
        
        print("Navigating to projections page...")
        driver.get('https://www.fangraphs.com/projections?pos=all&stats=bat&type=ratcdc')
        
        print("Waiting for page to load...")
        time.sleep(5)
        
        print("Current URL:", driver.current_url)
        
        # Download batters first
        download_projections_for_type(driver, "batters")
        
        # Then download pitchers
        print("Navigating to pitchers page...")
        driver.get('https://www.fangraphs.com/projections?type=ratcdc&stats=pit&pos=all&team=0&players=0&lg=all&z=1744973723&pageitems=30&statgroup=dashboard&fantasypreset=dashboard')
        time.sleep(5)  # Wait for page to load
        
        # Verify we're on the pitchers page
        if "stats=pit" not in driver.current_url:
            raise Exception("Failed to navigate to pitchers page")
        
        print("Looking for table...")
        # Wait for the table to be present
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