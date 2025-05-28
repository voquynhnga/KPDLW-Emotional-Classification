from collections import Counter
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

class DMXCrawler:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')  # không cần hiển thị giao diện
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-extensions')
        self.options.add_argument('--disable-software-rasterizer')  # tránh fallback WebGL
        self.options.add_argument('--remote-debugging-port=9222')
    
    def get_comments(self, url, max_pages=20):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        all_comments = []
        
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 5)
            
            try:
                view_all_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "c-btn-rate.btn-view-all")))
                view_all_button.click()
                time.sleep(1)
            except:
                print("View all comments button not found or not clickable")
            
            # Collect comments from multiple pages
            for page in range(max_pages):
                try:
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "cmt-txt")))
                    comment_elements = driver.find_elements(By.CSS_SELECTOR, "li[id^='r-']")
                    
                    for comment_el in comment_elements:
                        try:
                            comment_text_el = comment_el.find_element(By.CLASS_NAME, "cmt-txt")
                            comment_text = comment_text_el.text.strip()
                            
                            if comment_text and len(comment_text) > 10:
                                all_comments.append(comment_text)
                        except Exception as e:
                            continue
                    
                    # Try to go to next page
                    try:
                        current_page = driver.find_element(By.XPATH, "//div[@class='pagcomment']/span[@class='active']")
                        current_page_number = int(current_page.text)
                        next_page_number = current_page_number + 1
                        
                        next_page_link = driver.find_element(By.XPATH, f"//div[@class='pagcomment']/a[text()='{next_page_number}']")
                        driver.execute_script("arguments[0].click();", next_page_link)
                        time.sleep(2)
                    except:
                        print(f"Cannot navigate to page {page + 2}, stopping...")
                        break
                        
                except Exception as e:
                    print(f"Error on page {page + 1}: {e}")
                    break
            
        except Exception as e:
            print(f"Error crawling comments: {e}")
        finally:
            driver.quit()
        
        return list(set(all_comments))  # Remove duplicates
    
if __name__ == "__main__":
    crawler = DMXCrawler()
