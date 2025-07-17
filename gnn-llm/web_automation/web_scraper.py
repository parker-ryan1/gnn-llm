import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float
    url: str
    availability: bool
    description: str = ""
    rating: float = 0.0
    reviews_count: int = 0

@dataclass
class OrderRequest:
    products: List[Product]
    shipping_address: Dict[str, str]
    payment_info: Dict[str, str]
    user_credentials: Dict[str, str]

class WebScraper:
    def __init__(self, headless: bool = True):
        self.logger = logging.getLogger(__name__)
        self.headless = headless
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Chrome WebDriver with options"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    def search_products(self, query: str, site: str = "amazon") -> List[Product]:
        """Search for products on specified site"""
        if site.lower() == "amazon":
            return self._search_amazon(query)
        elif site.lower() == "ebay":
            return self._search_ebay(query)
        else:
            raise ValueError(f"Unsupported site: {site}")
    
    def _search_amazon(self, query: str) -> List[Product]:
        """Search Amazon for products"""
        products = []
        try:
            search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
            self.driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']"))
            )
            
            # Extract product information
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
            
            for element in product_elements[:10]:  # Limit to first 10 results
                try:
                    # Product name
                    name_elem = element.find_element(By.CSS_SELECTOR, "h2 a span")
                    name = name_elem.text.strip()
                    
                    # Product URL
                    url_elem = element.find_element(By.CSS_SELECTOR, "h2 a")
                    url = url_elem.get_attribute("href")
                    
                    # Price
                    price = 0.0
                    try:
                        price_elem = element.find_element(By.CSS_SELECTOR, ".a-price-whole")
                        price_text = price_elem.text.replace(",", "")
                        price = float(price_text)
                    except:
                        pass
                    
                    # Rating
                    rating = 0.0
                    try:
                        rating_elem = element.find_element(By.CSS_SELECTOR, ".a-icon-alt")
                        rating_text = rating_elem.get_attribute("textContent")
                        rating = float(rating_text.split()[0])
                    except:
                        pass
                    
                    # Availability (assume available if listed)
                    availability = True
                    
                    products.append(Product(
                        name=name,
                        price=price,
                        url=url,
                        availability=availability,
                        rating=rating
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting product info: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching Amazon: {e}")
        
        return products
    
    def _search_ebay(self, query: str) -> List[Product]:
        """Search eBay for products"""
        products = []
        try:
            search_url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
            self.driver.get(search_url)
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".s-item"))
            )
            
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, ".s-item")
            
            for element in product_elements[:10]:
                try:
                    # Skip ads
                    if "s-item--watch-at-corner" in element.get_attribute("class"):
                        continue
                    
                    # Product name
                    name_elem = element.find_element(By.CSS_SELECTOR, ".s-item__title")
                    name = name_elem.text.strip()
                    
                    # Product URL
                    url_elem = element.find_element(By.CSS_SELECTOR, ".s-item__link")
                    url = url_elem.get_attribute("href")
                    
                    # Price
                    price = 0.0
                    try:
                        price_elem = element.find_element(By.CSS_SELECTOR, ".s-item__price")
                        price_text = price_elem.text.replace("$", "").replace(",", "")
                        price = float(price_text.split()[0])
                    except:
                        pass
                    
                    products.append(Product(
                        name=name,
                        price=price,
                        url=url,
                        availability=True
                    ))
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching eBay: {e}")
        
        return products
    
    def get_product_details(self, product: Product) -> Product:
        """Get detailed information about a specific product"""
        try:
            self.driver.get(product.url)
            time.sleep(2)
            
            # Try to extract more details based on the site
            if "amazon.com" in product.url:
                return self._get_amazon_details(product)
            elif "ebay.com" in product.url:
                return self._get_ebay_details(product)
            
        except Exception as e:
            self.logger.error(f"Error getting product details: {e}")
        
        return product
    
    def _get_amazon_details(self, product: Product) -> Product:
        """Get detailed Amazon product information"""
        try:
            # Description
            try:
                desc_elem = self.driver.find_element(By.CSS_SELECTOR, "#feature-bullets ul")
                product.description = desc_elem.text.strip()
            except:
                pass
            
            # Reviews count
            try:
                reviews_elem = self.driver.find_element(By.CSS_SELECTOR, "#acrCustomerReviewText")
                reviews_text = reviews_elem.text
                product.reviews_count = int(reviews_text.split()[0].replace(",", ""))
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"Error getting Amazon details: {e}")
        
        return product
    
    def close(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
    
    def __del__(self):
        self.close()