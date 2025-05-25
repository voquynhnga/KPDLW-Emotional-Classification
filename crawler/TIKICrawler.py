import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

class TIKICrawler:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")  # Chạy không giao diện
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--window-size=1920x1080")
        self.options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

        self.index = 1
        self.stop_scraping = False
        self.last_page = False
        self.STOP_KEYWORD = "(*) Đánh giá không tính điểm"
    
    def scroll_down_slowly(self, step=500, delay=0.5):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        current_position = 0
        while current_position < last_height:
            self.driver.execute_script(f"window.scrollBy(0, {step});")
            time.sleep(delay)
            current_position += step
            last_height = self.driver.execute_script("return document.body.scrollHeight")

    def get_comments(self, url, max_pages=50):
        reviews = self.driver.find_elements(By.CSS_SELECTOR, ".review-comment")
        comments = []

        for review in reviews:
            if self.stop_scraping:
                break

            try:
                see_more = review.find_elements(By.CSS_SELECTOR, ".show-more-content")
                if see_more:
                    self.driver.execute_script("arguments[0].click();", see_more[0])
                    time.sleep(1)

                comment = review.find_element(By.CSS_SELECTOR, ".review-comment__content").text.strip()
                if not comment:
                    comment = review.find_element(By.CSS_SELECTOR, ".rating-attribute__attributes").text.strip()

                if self.STOP_KEYWORD in comment:
                    self.stop_scraping = True
                    break

                if comment:
                    print(f"{self.index}. {comment}")
                    print("-" * 50)
                    comments.append([self.index, comment])
                    self.index += 1
            except:
                continue
        
        return list(set(comments));

    def click_next_page(self):
        try:
            pagination = self.driver.find_elements(By.CSS_SELECTOR, ".customer-reviews__pagination li")
            if len(pagination) > 2:
                next_btn = pagination[-1]
                if "disabled" in next_btn.get_attribute("class"):
                    print("✅ Đã đến trang cuối.")
                    self.last_page = True
                    return False
                else:
                    ActionChains(self.driver).move_to_element(next_btn).click().perform()
                    time.sleep(2)
                    return True
        except:
            pass
        return False

    def crawl(self, url, csv_filename="reviews.csv", max_pages=50):
        self.driver.get(url)
        time.sleep(5)

        save_dir = "data"
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, csv_filename)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Comment"])

        self.scroll_down_slowly()

        page = 1
        while not self.stop_scraping and not self.last_page and page <= max_pages:
            reviews = self.get_reviews()
            with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(reviews)

            if not self.click_next_page():
                break

            page += 1

        self.driver.quit()

# --- Sử dụng ---
if __name__ == "__main__":
    crawler = TIKICrawler()
