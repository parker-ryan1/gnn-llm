import time
import json
import logging
from typing import Dict, List, Optional, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from .web_scraper import Product, OrderRequest
from dataclasses import dataclass

@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    total_cost: float = 0.0
    error_message: str = ""
    products_ordered: List[Product] = None

class OrderAutomation:
    def __init__(self, headless: bool = False):
        self.logger = logging.getLogger(__name__)
        self.headless = headless
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Chrome WebDriver for ordering"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Add user data directory to maintain sessions
        chrome_options.add_argument("--user-data-dir=/tmp/chrome_user_data")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    def login_to_site(self, site: str, credentials: Dict[str, str]) -> bool:
        """Login to the specified site"""
        try:
            if site.lower() == "amazon":
                return self._login_amazon(credentials)
            elif site.lower() == "ebay":
                return self._login_ebay(credentials)
            else:
                self.logger.error(f"Unsupported site for login: {site}")
                return False
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            return False
    
    def _login_amazon(self, credentials: Dict[str, str]) -> bool:
        """Login to Amazon"""
        try:
            self.driver.get("https://www.amazon.com/ap/signin")
            
            # Enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "ap_email"))
            )
            email_field.send_keys(credentials.get("email", ""))
            
            # Click continue
            continue_btn = self.driver.find_element(By.ID, "continue")
            continue_btn.click()
            
            # Enter password
            password_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "ap_password"))
            )
            password_field.send_keys(credentials.get("password", ""))
            
            # Click sign in
            signin_btn = self.driver.find_element(By.ID, "signInSubmit")
            signin_btn.click()
            
            # Wait for login to complete
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "nav-link-accountList"))
            )
            
            self.logger.info("Amazon login successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Amazon login failed: {e}")
            return False
    
    def _login_ebay(self, credentials: Dict[str, str]) -> bool:
        """Login to eBay"""
        try:
            self.driver.get("https://signin.ebay.com/")
            
            # Enter username
            username_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "userid"))
            )
            username_field.send_keys(credentials.get("username", ""))
            
            # Enter password
            password_field = self.driver.find_element(By.ID, "pass")
            password_field.send_keys(credentials.get("password", ""))
            
            # Click sign in
            signin_btn = self.driver.find_element(By.ID, "sgnBt")
            signin_btn.click()
            
            # Wait for login
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "gh-ug"))
            )
            
            self.logger.info("eBay login successful")
            return True
            
        except Exception as e:
            self.logger.error(f"eBay login failed: {e}")
            return False
    
    def add_to_cart(self, product: Product) -> bool:
        """Add a product to cart"""
        try:
            self.driver.get(product.url)
            time.sleep(2)
            
            if "amazon.com" in product.url:
                return self._add_to_cart_amazon()
            elif "ebay.com" in product.url:
                return self._add_to_cart_ebay()
            
        except Exception as e:
            self.logger.error(f"Failed to add product to cart: {e}")
            return False
    
    def _add_to_cart_amazon(self) -> bool:
        """Add Amazon product to cart"""
        try:
            # Find and click add to cart button
            add_to_cart_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "add-to-cart-button"))
            )
            add_to_cart_btn.click()
            
            # Handle any popups or additional options
            time.sleep(2)
            
            # Check if item was added successfully
            try:
                success_message = self.driver.find_element(By.CSS_SELECTOR, "#attachDisplayAddBaseAlert")
                if success_message:
                    self.logger.info("Product added to Amazon cart")
                    return True
            except:
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add to Amazon cart: {e}")
            return False
    
    def _add_to_cart_ebay(self) -> bool:
        """Add eBay product to cart"""
        try:
            # Find add to cart or buy it now button
            try:
                add_to_cart_btn = self.driver.find_element(By.ID, "atcBtn_btn_1")
                add_to_cart_btn.click()
            except:
                # Try buy it now if add to cart not available
                buy_now_btn = self.driver.find_element(By.ID, "binBtn_btn_1")
                buy_now_btn.click()
            
            time.sleep(2)
            self.logger.info("Product added to eBay cart")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add to eBay cart: {e}")
            return False
    
    def proceed_to_checkout(self, order_request: OrderRequest) -> OrderResult:
        """Proceed to checkout and complete the order"""
        try:
            # This is a DEMO implementation - DO NOT use with real payment info
            self.logger.warning("DEMO MODE: This is for demonstration only!")
            
            # Navigate to cart
            if "amazon.com" in self.driver.current_url:
                return self._checkout_amazon(order_request)
            elif "ebay.com" in self.driver.current_url:
                return self._checkout_ebay(order_request)
            
        except Exception as e:
            self.logger.error(f"Checkout failed: {e}")
            return OrderResult(success=False, error_message=str(e))
    
    def _checkout_amazon(self, order_request: OrderRequest) -> OrderResult:
        """DEMO: Amazon checkout process"""
        try:
            # Navigate to cart
            self.driver.get("https://www.amazon.com/gp/cart/view.html")
            
            # Click proceed to checkout
            checkout_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "proceedToRetailCheckout"))
            )
            checkout_btn.click()
            
            # DEMO: Stop here - don't actually complete the purchase
            self.logger.info("DEMO: Would proceed with Amazon checkout here")
            
            return OrderResult(
                success=True,
                order_id="DEMO-ORDER-123",
                total_cost=sum(p.price for p in order_request.products),
                products_ordered=order_request.products
            )
            
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))
    
    def _checkout_ebay(self, order_request: OrderRequest) -> OrderResult:
        """DEMO: eBay checkout process"""
        try:
            # Navigate to cart
            self.driver.get("https://cart.ebay.com/")
            
            # DEMO: Stop here - don't actually complete the purchase
            self.logger.info("DEMO: Would proceed with eBay checkout here")
            
            return OrderResult(
                success=True,
                order_id="DEMO-EBAY-456",
                total_cost=sum(p.price for p in order_request.products),
                products_ordered=order_request.products
            )
            
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))
    
    def close(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
    
    def __del__(self):
        self.close()