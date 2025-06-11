import time
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://43.142.162.35:5173/#/login"

class ResetPasswordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Chrome()
        cls.driver.maximize_window()
        cls.wait = WebDriverWait(cls.driver, 10)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_reset_password(self):
        driver = self.driver
        driver.get(BASE_URL)
        time.sleep(5)
        reset_link = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "忘记密码")]')))
        reset_link.click()
        time.sleep(2)
        email_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Email"]')))
        email_input.send_keys("2253729@tongji.edu.cn")
        new_pwd_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Password"]')))
        new_pwd_input.send_keys("123456")
        code_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "获取验证码")]')))
        code_btn.click()
        code = input("请输入邮箱收到的验证码：")
        code_input = self.wait.until(EC.visibility_of_element_located((By.XPATH, '//input[@placeholder="Email Code"]')))
        code_input.send_keys(code)
        submit_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, '//span[text()="确定"]')))
        submit_btn.click()
        self.wait.until(EC.visibility_of_element_located((By.XPATH, '//*[contains(text(), "密码重置成功")]')))

if __name__ == "__main__":
    unittest.main() 