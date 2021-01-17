import os
from appium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from multiprocessing import Process

os.system('adb connect 127.0.0.1:21503')

server = 'http://localhost:4723/wd/hub'
desired_caps = {
                "platformName": "Android",
                "deviceName": "127.0.0.1:21503",
                "appPackage": "com.eno.android.cj.page",
                "appActivity": "com.cjsc.cjeh.view.main.CJMainActivity",
                "noReset": "True"
                }
acc = '15502118293'
pd = '681116'

class Crawler():
    def __init__(self):
        self.driver = webdriver.Remote(server, desired_capabilities=desired_caps)
        self.wait = WebDriverWait(self.driver, 20)
        self.driver.implicitly_wait(2)
        self.login_success = self._login()


    def _login(self):
        try:
            td_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/cj_main_iv_trade')))
            td_button.click()
            login_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_center_login_btn')))
            login_button.click()
            password = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_login_password')))
            password.set_text(pd)
            login_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_login_login_btn')))
            login_button.click()
            self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_center_analysis')))
        except :
            self.close()
            return False
        else:
            return True

    def buy(self, code, amount):
        if not self.login_success:
            print('Login error')
            return False
        try:
            td_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/cj_main_iv_trade')))
            td_button.click()
            buy_button = self.wait.until(EC.presence_of_all_elements_located(
                (By.ID, 'com.eno.android.cj.page:id/item_trade_menu_action_icon')))[0]
            buy_button.click()
            code_input = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_base_stock_name')))
            code_input.set_text(code)
            limit = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_limit_up')))
            limit.click()
            amt_input = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_case_input')))
            amt_input.set_text(amount)
            buy_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_btn')))
            buy_button.click()
            confirm_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/hgt_confirm_definite')))
            confirm_button.click()
        except :
            self.close()
            return False
        else:
            return True

    def sell(self, code, amount):
        if not self.login_success:
            print('Login error')
            return False
        try:
            td_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/cj_main_iv_trade')))
            td_button.click()
            sell_button = self.wait.until(EC.presence_of_all_elements_located(
                (By.ID, 'com.eno.android.cj.page:id/item_trade_menu_action_icon')))[1]
            sell_button.click()
            code_input = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_base_stock_name')))
            code_input.set_text(code)
            limit = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_limit_down')))
            limit.click()
            amt_input = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_case_input')))
            amt_input.set_text(amount)
            buy_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/trade_buy_btn')))
            buy_button.click()
            confirm_button = self.wait.until(EC.presence_of_element_located(
                (By.ID, 'com.eno.android.cj.page:id/hgt_confirm_definite')))
            confirm_button.click()
        except :
            self.close()
            return False
        else:
            return True


    def close(self):
        self.driver.quit()

if __name__ == '__main__':
    driver = Crawler()
    driver.sell('600016', 2000)
    driver.close()