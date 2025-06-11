import time
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://43.142.162.35:5173/#/login"

class RegisterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Chrome()
        cls.driver.maximize_window()
        cls.wait = WebDriverWait(cls.driver, 10)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_register(self):
        driver = self.driver
        driver.get(BASE_URL)
        time.sleep(5)
        # reg_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//span[text()="注册"]')))
        # reg_btn.click()
        reg_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//span[text()="注册"]')))
        reg_btn.click()
        time.sleep(2)
        email_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Email"]')))
        email_input.send_keys("powderblue370@2925.com")
        username_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Username"]')))
        username_input.send_keys("powderblue370")
        password_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Password"]')))
        password_input.send_keys("123456")
        code_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "获取验证码")]')))
        code_btn.click()
        code = input("请输入邮箱收到的验证码：")
        code_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Email Code"]')))
        code_input.send_keys(code)
        submit_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@class="button login-submit"]')))
        submit_btn.click()
        self.wait.until(EC.visibility_of_element_located((By.XPATH, '//*[contains(text(), "注册成功")]')))

if __name__ == "__main__":
    unittest.main() 