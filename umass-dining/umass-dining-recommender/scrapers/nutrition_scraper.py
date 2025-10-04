import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class NutritionScraper:
    """
    Enhanced nutrition data scraper for UMass dining items
    Uses multiple data sources for comprehensive nutritional information
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Nutrition databases to try
        self.nutrition_sources = {
            'usda': 'https://fdc.nal.usda.gov/fdc-app.html#/',
            'nutritionix': 'https://www.nutritionix.com/api',
            'edamam': 'https://api.edamam.com/api/nutrition-data'
        }
        
        # Common nutrition patterns
        self.nutrition_patterns = {
            'calories': [r'(\d+)\s*cal', r'(\d+)\s*kcal', r'(\d+)\s*calories'],
            'protein': [r'(\d+(?:\.\d+)?)\s*g\s*protein', r'protein[:\s]*(\d+(?:\.\d+)?)'],
            'carbs': [r'(\d+(?:\.\d+)?)\s*g\s*carb', r'carb[:\s]*(\d+(?:\.\d+)?)'],
            'fat': [r'(\d+(?:\.\d+)?)\s*g\s*fat', r'fat[:\s]*(\d+(?:\.\d+)?)'],
            'fiber': [r'(\d+(?:\.\d+)?)\s*g\s*fiber', r'fiber[:\s]*(\d+(?:\.\d+)?)'],
            'sugar': [r'(\d+(?:\.\d+)?)\s*g\s*sugar', r'sugar[:\s]*(\d+(?:\.\d+)?)'],
            'sodium': [r'(\d+(?:\.\d+)?)\s*mg\s*sodium', r'sodium[:\s]*(\d+(?:\.\d+)?)']
        }
    
    def scrape_item_nutrition(self, item_name: str, description: str = "") -> Dict:
        """
        Scrape comprehensive nutrition data for a food item
        
        Args:
            item_name: Name of the food item
            description: Optional description for better matching
        
        Returns:
            Dictionary with nutrition information
        """
        nutrition_data = {
            'item_name': item_name,
            'scraped_at': datetime.now().isoformat(),
            'sources': [],
            'nutrition': {}
        }
        
        # Try multiple strategies
        strategies = [
            self._scrape_from_umass_nutrition,
            self._scrape_from_usda,
            self._scrape_from_general_web,
            self._estimate_from_description
        ]
        
        for strategy in strategies:
            try:
                result = strategy(item_name, description)
                if result and result.get('nutrition'):
                    nutrition_data['nutrition'].update(result['nutrition'])
                    nutrition_data['sources'].append(result.get('source', 'unknown'))
                    break
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        return nutrition_data
    
    def _scrape_from_umass_nutrition(self, item_name: str, description: str) -> Dict:
        """Try to get nutrition from UMass dining nutrition pages"""
        try:
            # UMass might have nutrition pages
            search_url = f"https://umassdining.com/search?q={item_name.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                nutrition = self._extract_nutrition_from_html(soup)
                
                if nutrition:
                    return {
                        'nutrition': nutrition,
                        'source': 'umass_dining'
                    }
        except Exception as e:
            logger.debug(f"UMass nutrition scrape failed: {e}")
        
        return {}
    
    def _scrape_from_usda(self, item_name: str, description: str) -> Dict:
        """Try to get nutrition from USDA database"""
        try:
            # This would require API key in real implementation
            # For now, return empty - would need to implement USDA API calls
            logger.debug("USDA scraping not implemented (requires API key)")
        except Exception as e:
            logger.debug(f"USDA scrape failed: {e}")
        
        return {}
    
    def _scrape_from_general_web(self, item_name: str, description: str) -> Dict:
        """Scrape nutrition from general web sources"""
        try:
            # Search for nutrition information
            search_terms = f"{item_name} nutrition facts calories"
            search_url = f"https://www.google.com/search?q={search_terms.replace(' ', '+')}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                nutrition = self._extract_nutrition_from_html(soup)
                
                if nutrition:
                    return {
                        'nutrition': nutrition,
                        'source': 'web_search'
                    }
        except Exception as e:
            logger.debug(f"Web nutrition scrape failed: {e}")
        
        return {}
    
    def _extract_nutrition_from_html(self, soup: BeautifulSoup) -> Dict:
        """Extract nutrition information from HTML content"""
        nutrition = {}
        text = soup.get_text().lower()
        
        for nutrient, patterns in self.nutrition_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        value = float(match.group(1))
                        nutrition[nutrient] = value
                        break
                    except ValueError:
                        continue
        
        return nutrition
    
    def _estimate_from_description(self, item_name: str, description: str) -> Dict:
        """Estimate nutrition based on food description and common patterns"""
        nutrition = {}
        text = f"{item_name} {description}".lower()
        
        # Calorie estimation based on food type
        calorie_estimates = {
            'salad': 50,
            'soup': 150,
            'pizza': 300,
            'pasta': 200,
            'chicken': 250,
            'beef': 300,
            'fish': 200,
            'vegetable': 30,
            'fruit': 60,
            'bread': 80,
            'rice': 150,
            'potato': 100,
            'cheese': 100,
            'dessert': 200,
            'ice cream': 150,
            'cake': 300,
            'cookie': 50
        }
        
        for food_type, calories in calorie_estimates.items():
            if food_type in text:
                nutrition['calories'] = calories
                break
        
        # Protein estimation
        if any(word in text for word in ['chicken', 'beef', 'fish', 'meat', 'protein']):
            nutrition['protein'] = 20
        elif any(word in text for word in ['cheese', 'milk', 'dairy']):
            nutrition['protein'] = 10
        elif any(word in text for word in ['bean', 'lentil', 'tofu']):
            nutrition['protein'] = 15
        
        # Fat estimation
        if any(word in text for word in ['fried', 'crispy', 'butter', 'oil', 'cheese']):
            nutrition['fat'] = 15
        elif any(word in text for word in ['grilled', 'baked', 'steamed']):
            nutrition['fat'] = 5
        
        # Carb estimation
        if any(word in text for word in ['pasta', 'rice', 'bread', 'potato', 'noodle']):
            nutrition['carbs'] = 40
        elif any(word in text for word in ['salad', 'vegetable', 'fruit']):
            nutrition['carbs'] = 10
        
        return {
            'nutrition': nutrition,
            'source': 'estimation'
        }
    
    def batch_scrape_nutrition(self, items: List[Dict]) -> List[Dict]:
        """
        Scrape nutrition for multiple items
        
        Args:
            items: List of item dictionaries with 'name' and optional 'description'
        
        Returns:
            List of nutrition data dictionaries
        """
        results = []
        
        for i, item in enumerate(items):
            logger.info(f"Scraping nutrition for item {i+1}/{len(items)}: {item['name']}")
            
            nutrition_data = self.scrape_item_nutrition(
                item['name'], 
                item.get('description', '')
            )
            
            results.append(nutrition_data)
            
            # Be respectful to servers
            time.sleep(1)
        
        return results
    
    def save_nutrition_data(self, nutrition_data: List[Dict], filename: str = None) -> str:
        """Save nutrition data to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/processed/nutrition_{timestamp}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'scraped_at': datetime.now().isoformat(),
                    'total_items': len(nutrition_data),
                    'nutrition_data': nutrition_data
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved nutrition data: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save nutrition data: {e}")
            return ""
    
    def load_nutrition_data(self, filename: str) -> List[Dict]:
        """Load nutrition data from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('nutrition_data', [])
        except Exception as e:
            logger.error(f"Failed to load nutrition data: {e}")
            return []
    
    def create_nutrition_database(self, menu_items: List[Dict]) -> Dict:
        """
        Create a comprehensive nutrition database from menu items
        
        Args:
            menu_items: List of menu items from menu scraper
        
        Returns:
            Dictionary with nutrition database
        """
        logger.info(f"Creating nutrition database for {len(menu_items)} items")
        
        # Get unique items
        unique_items = {}
        for item in menu_items:
            name = item['name'].lower().strip()
            if name not in unique_items:
                unique_items[name] = {
                    'name': item['name'],
                    'description': item.get('description', ''),
                    'frequency': 1
                }
            else:
                unique_items[name]['frequency'] += 1
        
        # Scrape nutrition for unique items
        items_to_scrape = list(unique_items.values())
        nutrition_results = self.batch_scrape_nutrition(items_to_scrape)
        
        # Create database
        nutrition_db = {
            'created_at': datetime.now().isoformat(),
            'total_items': len(nutrition_results),
            'items': {}
        }
        
        for result in nutrition_results:
            item_name = result['item_name'].lower().strip()
            nutrition_db['items'][item_name] = {
                'original_name': result['item_name'],
                'nutrition': result['nutrition'],
                'sources': result['sources'],
                'frequency': unique_items[item_name]['frequency']
            }
        
        return nutrition_db
    
    def get_nutrition_summary(self, nutrition_db: Dict) -> Dict:
        """Get summary statistics of nutrition database"""
        items = nutrition_db['items']
        
        summary = {
            'total_items': len(items),
            'items_with_calories': sum(1 for item in items.values() if 'calories' in item['nutrition']),
            'items_with_protein': sum(1 for item in items.values() if 'protein' in item['nutrition']),
            'items_with_carbs': sum(1 for item in items.values() if 'carbs' in item['nutrition']),
            'items_with_fat': sum(1 for item in items.values() if 'fat' in item['nutrition']),
            'avg_calories': 0,
            'avg_protein': 0,
            'avg_carbs': 0,
            'avg_fat': 0
        }
        
        # Calculate averages
        calories = [item['nutrition']['calories'] for item in items.values() if 'calories' in item['nutrition']]
        protein = [item['nutrition']['protein'] for item in items.values() if 'protein' in item['nutrition']]
        carbs = [item['nutrition']['carbs'] for item in items.values() if 'carbs' in item['nutrition']]
        fat = [item['nutrition']['fat'] for item in items.values() if 'fat' in item['nutrition']]
        
        if calories:
            summary['avg_calories'] = sum(calories) / len(calories)
        if protein:
            summary['avg_protein'] = sum(protein) / len(protein)
        if carbs:
            summary['avg_carbs'] = sum(carbs) / len(carbs)
        if fat:
            summary['avg_fat'] = sum(fat) / len(fat)
        
        return summary

# Example usage
if __name__ == "__main__":
    scraper = NutritionScraper()
    
    # Test with sample items
    test_items = [
        {'name': 'Chicken Breast', 'description': 'Grilled chicken breast'},
        {'name': 'Caesar Salad', 'description': 'Fresh romaine lettuce with caesar dressing'},
        {'name': 'Pizza Slice', 'description': 'Cheese pizza slice'},
        {'name': 'French Fries', 'description': 'Crispy golden french fries'}
    ]
    
    logger.info("Testing nutrition scraper...")
    results = scraper.batch_scrape_nutrition(test_items)
    
    for result in results:
        logger.info(f"\n{result['item_name']}:")
        for nutrient, value in result['nutrition'].items():
            logger.info(f"  {nutrient}: {value}")
    
    # Save results
    filename = scraper.save_nutrition_data(results)
    logger.info(f"Saved nutrition data to {filename}")
