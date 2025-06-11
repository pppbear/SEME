import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://43.142.162.35:5173/#/login"

class PredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Chrome()
        cls.driver.maximize_window()
        cls.wait = WebDriverWait(cls.driver, 10)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def login(self):
        driver = self.driver
        driver.get(BASE_URL)
        time.sleep(5)
        user_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Email/UserName"]')))
        user_input.send_keys("xyy")
        pwd_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Password"]')))
        pwd_input.send_keys("123456")
        login_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//*[contains(text(), "登录")]')))
        login_btn.click()
        self.wait.until(EC.url_contains("/grid"))
        time.sleep(5)

    def test_predict(self):
        driver = self.driver
        self.login()
        driver.get("http://43.142.162.35:5173/#/predict")
        upload_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
        upload_input.send_keys(r"D:\vscode\project\test\system_level\test_sample.xlsx")
        select = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'el-select')))
        select.click()
        option = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "白天地表平均温度")]')))
        option.click()
        predict_btn = self.wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[normalize-space(.)="预测"]'))
        )
        predict_btn.click()
        self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'chart')))
        time.sleep(5)

if __name__ == "__main__":
    unittest.main() 