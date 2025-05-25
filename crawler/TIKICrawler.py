from collections import Counter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time

class TIKICrawler:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--window-size=1920,1080')
        self.STOP_KEYWORD = "(*) Đánh giá không tính điểm"
   
    def get_comments(self, url, max_pages=20):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        all_comments = []
        stop_scraping = False
        
        try:
            driver.get(url)
            time.sleep(5)
            wait = WebDriverWait(driver, 5)
            
            # Scroll down to load content
            self._scroll_down_slowly(driver)
            
            # Collect comments from multiple pages
            for page in range(max_pages):
                if stop_scraping:
                    break
                    
                try:
                    # Get reviews from current page
                    page_comments, stop_scraping = self._get_reviews_from_page(driver)
                    all_comments.extend(page_comments)
                    
                    if stop_scraping:
                        print("Stop keyword found, ending scraping...")
                        break
                    
                    # Try to go to next page
                    if not self._click_next_page(driver):
                        print(f"Cannot navigate to next page, stopping...")
                        break
                        
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error on page {page + 1}: {e}")
                    break
           
        except Exception as e:
            print(f"Error crawling comments: {e}")
        finally:
            driver.quit()
       
        return list(set(all_comments))  # Remove duplicates
    
    def _scroll_down_slowly(self, driver, step=500, delay=0.5):
        """Scroll down slowly to load content"""
        last_height = driver.execute_script("return document.body.scrollHeight")
        current_position = 0
        
        while current_position < last_height:
            driver.execute_script(f"window.scrollBy(0, {step});")
            time.sleep(delay)
            current_position += step
            last_height = driver.execute_script("return document.body.scrollHeight")
    
    def _get_reviews_from_page(self, driver):
        """Extract reviews from current page"""
        reviews = driver.find_elements(By.CSS_SELECTOR, ".review-comment")
        page_comments = []
        stop_scraping = False
        
        for review in reviews:
            if stop_scraping:
                break
                
            try:
                # Click "see more" button if exists
                see_more_button = review.find_elements(By.CSS_SELECTOR, ".show-more-content")
                if see_more_button:
                    driver.execute_script("arguments[0].click();", see_more_button[0])
                    time.sleep(1)
                
                # Extract comment text
                comment = ""
                try:
                    comment = review.find_element(By.CSS_SELECTOR, ".review-comment__content").text.strip()
                except:
                    try:
                        comment = review.find_element(By.CSS_SELECTOR, ".rating-attribute__attributes").text.strip()
                    except:
                        continue
                
                # Check for stop keyword
                if self.STOP_KEYWORD in comment:
                    stop_scraping = True
                    break
                
                # Add valid comments
                if comment and len(comment) > 10:
                    page_comments.append(comment)
                    
            except Exception as e:
                continue
        
        return page_comments, stop_scraping
    
    def _click_next_page(self, driver):
        """Navigate to next page"""
        try:
            pagination = driver.find_elements(By.CSS_SELECTOR, ".customer-reviews__pagination li")
            
            if len(pagination) > 2:
                next_button = pagination[-1]  # Last element should be next button
                
                if "disabled" in next_button.get_attribute("class"):
                    print("✅ Reached last page, stopping data collection!")
                    return False
                else:
                    ActionChains(driver).move_to_element(next_button).click().perform()
                    return True
            
        except Exception as e:
            print(f"Error clicking next page: {e}")
            return False
        
        return False

if __name__ == "__main__":
    crawler = TIKICrawler()
