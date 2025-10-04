import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Any
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UMassDiningScraper:
    """
    Enhanced UMass Dining Menu Scraper with better error handling and data extraction
    """
    
    def __init__(self):
        self.base_url = "https://umassdining.com"
        self.dining_halls = {
            'worcester': 'Worcester Dining Commons',
            'franklin': 'Franklin Dining Commons', 
            'berkshire': 'Berkshire Dining Commons',
            'hampshire': 'Hampshire Dining Commons'
        }
        
        # Create data directories
        Path("data/raw/menus").mkdir(parents=True, exist_ok=True)
        
        # Enhanced session with better headers and retry logic
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (XHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP {response.status_code} for {url}")
                    if attempt == self.max_retries - 1:
                        return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay)
        
        return None
    
    def scrape_daily_menu(self, dining_hall: str, date: datetime = None) -> Optional[Dict]:
        """
        Scrape menu for a specific dining hall and date
        
        Args:
            dining_hall: 'worcester', 'franklin', 'berkshire', 'hampshire'
            date: datetime object (default: today)
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        url = f"{self.base_url}/locations-menus/{dining_hall}/menu"
        
        logger.info(f"Scraping {dining_hall} for {date_str}")
        
        try:
            # Try different URL patterns
            urls_to_try = [
                f"{self.base_url}/locations-menus/{dining_hall}/menu",
                f"{self.base_url}/locations-menus/{dining_hall}",
                f"{self.base_url}/menu/{dining_hall}",
            ]
            
            response = None
            for url in urls_to_try:
                response = self._make_request(url, {'date': date_str})
                if response:
                    break
            
            if not response:
                logger.error(f"Failed to get response for {dining_hall}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            menu_data = {
                'dining_hall': dining_hall,
                'date': date_str,
                'meals': {},
                'scraped_at': datetime.now().isoformat()
            }
            
            # Try multiple parsing strategies
            meals = self._parse_meals_multiple_strategies(soup)
            menu_data['meals'] = meals
            
            if not meals:
                logger.warning(f"No meals found for {dining_hall} on {date_str}")
                # Try alternative parsing
                menu_data['meals'] = self._parse_alternative_structure(soup)
            
            return menu_data
            
        except Exception as e:
            logger.error(f"Exception scraping {dining_hall}: {e}")
            return None
    
    def _parse_meals_multiple_strategies(self, soup: BeautifulSoup) -> Dict:
        """Try multiple strategies to parse meal data"""
        meals = {}
        
        # Strategy 1: Look for common meal period patterns
        meal_patterns = [
            {'class': 'menu-period'},
            {'class': 'meal-period'},
            {'class': 'period'},
            {'class': 'meal'},
            {'id': re.compile(r'(breakfast|lunch|dinner)', re.I)},
        ]
        
        for pattern in meal_patterns:
            meal_elements = soup.find_all('div', pattern)
            if meal_elements:
                meals = self._parse_meal_elements(meal_elements)
                if meals:
                    break
        
        # Strategy 2: Look for time-based patterns
        if not meals:
            time_patterns = soup.find_all(text=re.compile(r'(breakfast|lunch|dinner|brunch)', re.I))
            if time_patterns:
                meals = self._parse_time_based_meals(soup, time_patterns)
        
        return meals
    
    def _parse_meal_elements(self, meal_elements: List) -> Dict:
        """Parse meal elements into structured data"""
        meals = {}
        
        for period in meal_elements:
            # Get meal name
            meal_name_elem = period.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not meal_name_elem:
                meal_name_elem = period.find(['span', 'div'], class_=re.compile(r'title|name|header', re.I))
            
            if not meal_name_elem:
                continue
                
            meal_name = meal_name_elem.get_text().strip().lower()
            meal_name = re.sub(r'[^\w\s]', '', meal_name)
            
            # Parse stations
            stations = self._parse_stations(period)
            if stations:
                meals[meal_name] = stations
        
        return meals
    
    def _parse_stations(self, meal_element) -> Dict:
        """Parse food stations within a meal"""
        stations = {}
        
        # Look for station patterns
        station_patterns = [
            {'class': 'menu-station'},
            {'class': 'station'},
            {'class': 'food-station'},
            {'class': 'category'},
        ]
        
        station_elements = []
        for pattern in station_patterns:
            station_elements = meal_element.find_all('div', pattern)
            if station_elements:
                break
        
        if not station_elements:
            # Fallback: look for any div with food items
            station_elements = meal_element.find_all('div', class_=re.compile(r'item|food|menu', re.I))
        
        for station in station_elements:
            station_name = self._extract_station_name(station)
            if not station_name:
                continue
                
            items = self._parse_menu_items(station)
            if items:
                stations[station_name] = items
        
        return stations
    
    def _extract_station_name(self, station_element) -> str:
        """Extract station name from element"""
        # Try different selectors for station name
        name_selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            '.title', '.name', '.header', '.station-name',
            'span', 'div'
        ]
        
        for selector in name_selectors:
            name_elem = station_element.find(selector)
            if name_elem and name_elem.get_text().strip():
                return name_elem.get_text().strip()
        
        return "Unknown Station"
    
    def _parse_menu_items(self, station_element) -> List[Dict]:
        """Parse individual menu items from a station"""
        items = []
        
        # Look for item patterns
        item_patterns = [
            {'class': 'menu-item'},
            {'class': 'item'},
            {'class': 'food-item'},
            {'class': 'menu-entry'},
        ]
        
        item_elements = []
        for pattern in item_patterns:
            item_elements = station_element.find_all('div', pattern)
            if item_elements:
                break
        
        if not item_elements:
            # Fallback: look for any element that might contain food names
            item_elements = station_element.find_all(['li', 'span', 'div'], 
                                                   text=re.compile(r'[a-zA-Z]'))
        
        for item in item_elements:
            item_data = self._extract_item_data(item)
            if item_data:
                items.append(item_data)
        
        return items
    
    def _extract_item_data(self, item_element) -> Optional[Dict]:
        """Extract data from a menu item element"""
        # Get item name
        item_name = self._extract_item_name(item_element)
        if not item_name or len(item_name) < 2:
            return None
        
        # Extract additional data
        nutrition = self._parse_nutrition(item_element)
        allergens = self._parse_allergens(item_element)
        description = self._get_item_description(item_element)
        
        return {
            'name': item_name,
            'nutrition': nutrition,
            'allergens': allergens,
            'description': description,
            'raw_html': str(item_element)[:500]  # Keep some raw HTML for debugging
        }
    
    def _extract_item_name(self, item_element) -> str:
        """Extract item name from element"""
        # Try different strategies
        name_selectors = [
            '.item-name', '.name', '.title', '.food-name',
            'span', 'div', 'a'
        ]
        
        for selector in name_selectors:
            name_elem = item_element.find(selector)
            if name_elem:
                text = name_elem.get_text().strip()
                if text and len(text) > 1:
                    return text
        
        # Fallback: use element text directly
        text = item_element.get_text().strip()
        if text and len(text) < 100:  # Reasonable length for food name
            return text
        
        return ""
    
    def _parse_nutrition(self, item_element) -> Dict:
        """Extract nutritional information with multiple strategies"""
        nutrition = {}
        
        # Strategy 1: Look for nutrition facts div
        nutrition_div = item_element.find('div', class_=re.compile(r'nutrition', re.I))
        if nutrition_div:
            nutrition = self._extract_nutrition_from_div(nutrition_div)
        
        # Strategy 2: Look for calorie information in text
        if not nutrition:
            text = item_element.get_text()
            calorie_match = re.search(r'(\d+)\s*cal', text, re.I)
            if calorie_match:
                nutrition['calories'] = calorie_match.group(1)
        
        return nutrition
    
    def _extract_nutrition_from_div(self, nutrition_div) -> Dict:
        """Extract nutrition data from a nutrition div"""
        nutrition = {}
        
        # Look for common nutrition labels
        nutrition_labels = {
            'calories': ['cal', 'calorie'],
            'protein': ['protein', 'prot'],
            'carbs': ['carbs', 'carbohydrate'],
            'fat': ['fat', 'lipid'],
            'fiber': ['fiber', 'fibre'],
            'sugar': ['sugar', 'sugars'],
            'sodium': ['sodium', 'salt']
        }
        
        text = nutrition_div.get_text().lower()
        
        for nutrient, labels in nutrition_labels.items():
            for label in labels:
                pattern = rf'(\d+(?:\.\d+)?)\s*{label}'
                match = re.search(pattern, text)
                if match:
                    nutrition[nutrient] = match.group(1)
                    break
        
        return nutrition
    
    def _parse_allergens(self, item_element) -> List[str]:
        """Extract allergen information with multiple strategies"""
        allergens = []
        
        # Strategy 1: Look for allergen icons
        allergen_icons = item_element.find_all('img', class_=re.compile(r'allergen', re.I))
        for icon in allergen_icons:
            alt_text = icon.get('alt', '').strip()
            if alt_text:
                allergens.append(alt_text)
        
        # Strategy 2: Look for allergen text
        text = item_element.get_text().lower()
        allergen_keywords = [
            'gluten', 'dairy', 'nuts', 'peanuts', 'soy', 'eggs',
            'shellfish', 'fish', 'sesame', 'vegan', 'vegetarian'
        ]
        
        for allergen in allergen_keywords:
            if allergen in text:
                allergens.append(allergen)
        
        return list(set(allergens))  # Remove duplicates
    
    def _get_item_description(self, item_element) -> str:
        """Get item description if available"""
        desc_selectors = [
            '.description', '.desc', '.item-description',
            'p', '.text', '.details'
        ]
        
        for selector in desc_selectors:
            desc_elem = item_element.find(selector)
            if desc_elem:
                text = desc_elem.get_text().strip()
                if text and len(text) > 5:
                    return text
        
        return ""
    
    def _parse_alternative_structure(self, soup: BeautifulSoup) -> Dict:
        """Fallback parsing for different website structures"""
        meals = {}
        
        # Look for any text that might indicate meal periods
        all_text = soup.get_text().lower()
        
        meal_periods = ['breakfast', 'lunch', 'dinner', 'brunch']
        for period in meal_periods:
            if period in all_text:
                meals[period] = {
                    'General': [{'name': f'Items from {period}', 'nutrition': {}, 'allergens': [], 'description': ''}]
                }
        
        return meals
    
    def _parse_time_based_meals(self, soup: BeautifulSoup, time_patterns: List) -> Dict:
        """Parse meals based on time patterns found in text"""
        meals = {}
        
        for pattern in time_patterns:
            parent = pattern.parent
            if parent:
                meal_name = pattern.strip().lower()
                meal_name = re.sub(r'[^\w\s]', '', meal_name)
                
                # Look for nearby food items
                items = self._find_nearby_items(parent)
                if items:
                    meals[meal_name] = {'General': items}
        
        return meals
    
    def _find_nearby_items(self, element) -> List[Dict]:
        """Find food items near a given element"""
        items = []
        
        # Look in siblings and children
        for sibling in element.find_next_siblings():
            item_data = self._extract_item_data(sibling)
            if item_data:
                items.append(item_data)
        
        for child in element.find_all(['div', 'li', 'span']):
            item_data = self._extract_item_data(child)
            if item_data:
                items.append(item_data)
        
        return items[:10]  # Limit to avoid too many items
    
    def scrape_all_dining_halls(self, date: datetime = None) -> Dict:
        """Scrape all dining halls for a given date"""
        all_menus = {}
        
        for hall_id, hall_name in self.dining_halls.items():
            logger.info(f"Scraping {hall_name}...")
            menu = self.scrape_daily_menu(hall_id, date)
            
            if menu:
                all_menus[hall_id] = menu
                logger.info(f"Successfully scraped {hall_name}")
            else:
                logger.warning(f"Failed to scrape {hall_name}")
            
            # Be respectful - don't hammer the server
            time.sleep(2)
        
        return all_menus
    
    def save_menu_data(self, menu_data: Dict, date: datetime = None) -> str:
        """Save scraped menu data with better file naming"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y%m%d')
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"data/raw/menus/menu_{date_str}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': date.strftime('%Y-%m-%d'),
                    'scraped_at': datetime.now().isoformat(),
                    'menus': menu_data,
                    'scraper_version': '2.0'
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved menu data: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save menu data: {e}")
            return ""
    
    def scrape_historical_menus(self, days_back: int = 30) -> List[str]:
        """Scrape menus for the past N days"""
        today = datetime.now()
        saved_files = []
        
        for i in range(days_back):
            date = today - timedelta(days=i)
            logger.info(f"Scraping {date.strftime('%Y-%m-%d')}...")
            
            menus = self.scrape_all_dining_halls(date)
            if menus:
                filename = self.save_menu_data(menus, date)
                if filename:
                    saved_files.append(filename)
            
            # Don't overwhelm server
            time.sleep(5)
        
        logger.info(f"Scraped {len(saved_files)} days of historical data")
        return saved_files
    
    def get_available_dates(self) -> List[str]:
        """Check what dates have menu data available"""
        menu_files = list(Path("data/raw/menus").glob("menu_*.json"))
        dates = []
        
        for file_path in menu_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    dates.append(data['date'])
            except:
                continue
        
        return sorted(set(dates))

# Example usage and testing
if __name__ == "__main__":
    scraper = UMassDiningScraper()
    
    # Test scraping today's menu
    logger.info("Testing scraper with today's menu...")
    menus = scraper.scrape_all_dining_halls()
    
    if menus:
        filename = scraper.save_menu_data(menus)
        logger.info(f"Successfully saved menu data to {filename}")
        
        # Print sample data
        for hall, menu in menus.items():
            logger.info(f"\n{hall.upper()} DINING HALL:")
            for meal, stations in menu['meals'].items():
                logger.info(f"  {meal.upper()}:")
                for station, items in stations.items():
                    logger.info(f"    {station}: {len(items)} items")
    else:
        logger.error("Failed to scrape any menu data")
    
    # Check available dates
    available_dates = scraper.get_available_dates()
    logger.info(f"Available menu dates: {available_dates}")
