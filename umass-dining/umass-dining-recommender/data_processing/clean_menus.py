import json
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class MenuProcessor:
    """
    Enhanced menu data processor with advanced cleaning and standardization
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw" / "menus"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Common food item variations to standardize
        self.item_mappings = {
            'chicken breast': ['chicken', 'chx breast', 'grilled chicken', 'chicken breast grilled'],
            'french fries': ['fries', 'ff', 'french fry', 'potato fries'],
            'pizza': ['cheese pizza', 'pepperoni pizza', 'za', 'pizza slice'],
            'caesar salad': ['caesar', 'caesar salad', 'romaine salad'],
            'grilled salmon': ['salmon', 'grilled fish', 'salmon fillet'],
            'pasta': ['spaghetti', 'noodles', 'pasta dish'],
            'rice': ['white rice', 'brown rice', 'steamed rice'],
            'soup': ['soup of the day', 'daily soup', 'vegetable soup'],
            'sandwich': ['sandwich', 'sub', 'wrap', 'panini'],
            'burger': ['hamburger', 'cheeseburger', 'veggie burger']
        }
        
        # Station mappings
        self.station_mappings = {
            'international': ['international', 'world cuisine', 'global'],
            'grill': ['grill', 'grilled', 'bbq', 'barbecue'],
            'pizza': ['pizza', 'italian', 'pasta'],
            'salad': ['salad bar', 'salad', 'fresh', 'greens'],
            'vegetarian': ['vegetarian', 'vegan', 'plant based'],
            'dessert': ['dessert', 'sweets', 'treats'],
            'soup': ['soup', 'soups', 'broth']
        }
        
        # Allergen mappings
        self.allergen_mappings = {
            'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt'],
            'gluten': ['wheat', 'flour', 'bread', 'pasta'],
            'nuts': ['peanuts', 'almonds', 'walnuts', 'cashews'],
            'eggs': ['egg', 'eggs'],
            'soy': ['soy', 'soybean', 'tofu'],
            'fish': ['fish', 'salmon', 'tuna', 'seafood'],
            'shellfish': ['shrimp', 'crab', 'lobster', 'shellfish']
        }
    
    def load_all_menus(self) -> List[Dict[str, Any]]:
        """Load all scraped menu files"""
        all_menus = []
        
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory not found: {self.raw_data_dir}")
            return all_menus
        
        for file_path in self.raw_data_dir.glob("menu_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_menus.append(data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_menus)} menu files")
        return all_menus
    
    def flatten_menu_to_items(self, menu_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert nested menu structure to flat list of items
        
        Returns: List of dicts with item details
        """
        items = []
        
        try:
            date = menu_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            for dining_hall, menu in menu_data.get('menus', {}).items():
                for meal_period, stations in menu.get('meals', {}).items():
                    for station_name, station_items in stations.items():
                        for item in station_items:
                            # Clean and process item
                            cleaned_item = self._clean_item_data(item)
                            
                            if cleaned_item:
                                items.append({
                                    'date': date,
                                    'dining_hall': dining_hall,
                                    'meal_period': meal_period,
                                    'station': station_name,
                                    'station_clean': self._clean_station_name(station_name),
                                    'item_name': item.get('name', ''),
                                    'item_name_clean': cleaned_item['name_clean'],
                                    'description': cleaned_item.get('description', ''),
                                    'calories': cleaned_item.get('calories'),
                                    'protein': cleaned_item.get('protein'),
                                    'carbs': cleaned_item.get('carbs'),
                                    'fat': cleaned_item.get('fat'),
                                    'fiber': cleaned_item.get('fiber'),
                                    'sugar': cleaned_item.get('sugar'),
                                    'sodium': cleaned_item.get('sodium'),
                                    'allergens': cleaned_item.get('allergens', []),
                                    'allergens_text': ','.join(cleaned_item.get('allergens', [])),
                                    'is_vegan': cleaned_item.get('is_vegan', False),
                                    'is_vegetarian': cleaned_item.get('is_vegetarian', False),
                                    'is_gluten_free': cleaned_item.get('is_gluten_free', False),
                                    'has_dairy': cleaned_item.get('has_dairy', False),
                                    'has_nuts': cleaned_item.get('has_nuts', False),
                                    'cuisine_type': self._infer_cuisine_type(cleaned_item['name_clean']),
                                    'meal_type': self._infer_meal_type(cleaned_item['name_clean']),
                                    'health_score': self._calculate_health_score(cleaned_item)
                                })
        except Exception as e:
            logger.error(f"Error flattening menu data: {e}")
        
        return items
    
    def _clean_item_data(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize item data"""
        try:
            name = item.get('name', '').strip()
            if not name or len(name) < 2:
                return None
            
            # Clean name
            name_clean = self._clean_item_name(name)
            
            # Extract nutrition data
            nutrition = item.get('nutrition', {})
            cleaned_nutrition = self._clean_nutrition_data(nutrition)
            
            # Extract allergens
            allergens = item.get('allergens', [])
            cleaned_allergens = self._clean_allergens(allergens)
            
            # Extract description
            description = item.get('description', '').strip()
            
            return {
                'name_clean': name_clean,
                'description': description,
                'allergens': cleaned_allergens,
                **cleaned_nutrition
            }
        except Exception as e:
            logger.error(f"Error cleaning item data: {e}")
            return None
    
    def _clean_item_name(self, name: str) -> str:
        """Standardize item names"""
        # Lowercase and strip
        name = name.lower().strip()
        
        # Remove special characters but keep spaces and hyphens
        name = re.sub(r'[^\w\s-]', '', name)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Apply mappings
        for standard_name, variations in self.item_mappings.items():
            if any(var in name for var in variations):
                return standard_name
        
        return name
    
    def _clean_station_name(self, station: str) -> str:
        """Standardize station names"""
        station_lower = station.lower().strip()
        
        for standard_name, variations in self.station_mappings.items():
            if any(var in station_lower for var in variations):
                return standard_name
        
        return station_lower
    
    def _clean_nutrition_data(self, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize nutrition data"""
        cleaned = {}
        
        for nutrient, value in nutrition.items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Extract numeric value
                if isinstance(value, str):
                    # Remove units and extract number
                    numeric_value = re.findall(r'[\d.]+', value)
                    if numeric_value:
                        cleaned[nutrient] = float(numeric_value[0])
                else:
                    cleaned[nutrient] = float(value)
            except (ValueError, TypeError):
                continue
        
        return cleaned
    
    def _clean_allergens(self, allergens: List[str]) -> List[str]:
        """Clean and standardize allergen information"""
        cleaned = []
        
        for allergen in allergens:
            if not allergen or pd.isna(allergen):
                continue
            
            allergen_lower = allergen.lower().strip()
            
            # Map to standard allergen names
            for standard_allergen, variations in self.allergen_mappings.items():
                if any(var in allergen_lower for var in variations):
                    if standard_allergen not in cleaned:
                        cleaned.append(standard_allergen)
                    break
            else:
                # Keep original if no mapping found
                if allergen_lower not in cleaned:
                    cleaned.append(allergen_lower)
        
        return cleaned
    
    def _infer_cuisine_type(self, item_name: str) -> str:
        """Infer cuisine type from item name"""
        cuisine_indicators = {
            'italian': ['pasta', 'pizza', 'marinara', 'parmesan', 'mozzarella', 'italian'],
            'mexican': ['taco', 'burrito', 'salsa', 'guacamole', 'jalapeno', 'mexican'],
            'asian': ['rice', 'noodle', 'soy', 'ginger', 'sesame', 'teriyaki', 'asian'],
            'indian': ['curry', 'tikka', 'masala', 'naan', 'dal', 'indian'],
            'american': ['burger', 'fries', 'cheese', 'bacon', 'grilled', 'american'],
            'mediterranean': ['hummus', 'olive', 'feta', 'tzatziki', 'mediterranean'],
            'chinese': ['lo mein', 'fried rice', 'chow mein', 'chinese'],
            'thai': ['pad thai', 'thai', 'coconut', 'lemongrass']
        }
        
        for cuisine, indicators in cuisine_indicators.items():
            if any(indicator in item_name for indicator in indicators):
                return cuisine
        
        return 'other'
    
    def _infer_meal_type(self, item_name: str) -> str:
        """Infer meal type from item name"""
        meal_indicators = {
            'breakfast': ['egg', 'pancake', 'waffle', 'cereal', 'toast', 'bacon', 'bagel'],
            'lunch': ['sandwich', 'wrap', 'salad', 'soup', 'pizza'],
            'dinner': ['pasta', 'rice', 'meat', 'chicken', 'beef', 'fish', 'curry'],
            'dessert': ['cake', 'pie', 'ice cream', 'cookie', 'brownie', 'pudding'],
            'snack': ['chips', 'crackers', 'nuts', 'fruit', 'yogurt']
        }
        
        for meal_type, indicators in meal_indicators.items():
            if any(indicator in item_name for indicator in indicators):
                return meal_type
        
        return 'main'
    
    def _calculate_health_score(self, item_data: Dict[str, Any]) -> int:
        """Calculate health score for an item (0-10)"""
        score = 5  # Base score
        
        # Nutrition-based scoring
        calories = item_data.get('calories', 0)
        protein = item_data.get('protein', 0)
        fat = item_data.get('fat', 0)
        fiber = item_data.get('fiber', 0)
        sugar = item_data.get('sugar', 0)
        sodium = item_data.get('sodium', 0)
        
        if calories > 0:
            if calories < 200:
                score += 2
            elif calories < 400:
                score += 1
            elif calories > 600:
                score -= 2
            elif calories > 800:
                score -= 3
        
        if protein > 20:
            score += 1
        elif protein > 15:
            score += 0.5
        
        if fat < 10:
            score += 1
        elif fat > 20:
            score -= 1
        
        if fiber > 5:
            score += 1
        
        if sugar > 20:
            score -= 1
        
        if sodium > 800:
            score -= 1
        
        # Dietary preference scoring
        if item_data.get('is_vegan'):
            score += 1
        if item_data.get('is_vegetarian'):
            score += 0.5
        if item_data.get('is_gluten_free'):
            score += 0.5
        
        # Clamp to 0-10 range
        return max(0, min(10, int(score)))
    
    def create_item_database(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a master database of all unique food items
        """
        all_menus = self.load_all_menus()
        all_items = []
        
        for menu in all_menus:
            items = self.flatten_menu_to_items(menu)
            all_items.extend(items)
        
        if not all_items:
            logger.warning("No items found to process")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_items)
        
        # Save raw items
        raw_items_file = self.processed_dir / "all_menu_items.csv"
        df.to_csv(raw_items_file, index=False)
        logger.info(f"Saved {len(df)} menu items to {raw_items_file}")
        
        # Create unique items database
        unique_items = df.groupby('item_name_clean').agg({
            'item_name': 'first',  # Keep original name
            'description': lambda x: x.mode()[0] if len(x) > 0 else '',
            'calories': 'mean',
            'protein': 'mean',
            'carbs': 'mean',
            'fat': 'mean',
            'fiber': 'mean',
            'sugar': 'mean',
            'sodium': 'mean',
            'allergens_text': lambda x: ','.join(set(','.join(x).split(','))),
            'station_clean': lambda x: x.mode()[0] if len(x) > 0 else '',
            'cuisine_type': lambda x: x.mode()[0] if len(x) > 0 else 'other',
            'meal_type': lambda x: x.mode()[0] if len(x) > 0 else 'main',
            'health_score': 'mean',
            'is_vegan': 'any',
            'is_vegetarian': 'any',
            'is_gluten_free': 'any',
            'has_dairy': 'any',
            'has_nuts': 'any',
            'date': 'count'  # Frequency
        }).reset_index()
        
        unique_items.rename(columns={'date': 'frequency'}, inplace=True)
        
        # Add item ID
        unique_items['item_id'] = range(len(unique_items))
        
        # Reorder columns
        column_order = [
            'item_id', 'item_name_clean', 'item_name', 'description',
            'calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'sodium',
            'allergens_text', 'station_clean', 'cuisine_type', 'meal_type',
            'health_score', 'is_vegan', 'is_vegetarian', 'is_gluten_free',
            'has_dairy', 'has_nuts', 'frequency'
        ]
        
        unique_items = unique_items[column_order]
        
        # Save unique items
        unique_items_file = self.processed_dir / "unique_items.csv"
        unique_items.to_csv(unique_items_file, index=False)
        logger.info(f"Created database of {len(unique_items)} unique items")
        
        return df, unique_items
    
    def analyze_menu_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find interesting patterns in the data"""
        analysis = {}
        
        try:
            # Most common items
            analysis['most_common_items'] = df['item_name_clean'].value_counts().head(10).to_dict()
            
            # Items by dining hall
            analysis['items_by_dining_hall'] = df.groupby('dining_hall')['item_name'].count().to_dict()
            
            # Meal distribution
            analysis['meal_distribution'] = df.groupby('meal_period')['item_name'].count().to_dict()
            
            # Station popularity
            analysis['station_popularity'] = df['station_clean'].value_counts().head(10).to_dict()
            
            # Cuisine distribution
            analysis['cuisine_distribution'] = df['cuisine_type'].value_counts().to_dict()
            
            # Health score distribution
            analysis['health_score_stats'] = {
                'mean': df['health_score'].mean(),
                'std': df['health_score'].std(),
                'min': df['health_score'].min(),
                'max': df['health_score'].max()
            }
            
            # Nutritional analysis
            nutritional_stats = {}
            for nutrient in ['calories', 'protein', 'carbs', 'fat']:
                if nutrient in df.columns:
                    nutritional_stats[nutrient] = {
                        'mean': df[nutrient].mean(),
                        'median': df[nutrient].median(),
                        'std': df[nutrient].std()
                    }
            analysis['nutritional_stats'] = nutritional_stats
            
            # Dietary preferences
            analysis['dietary_preferences'] = {
                'vegan_items': df['is_vegan'].sum(),
                'vegetarian_items': df['is_vegetarian'].sum(),
                'gluten_free_items': df['is_gluten_free'].sum(),
                'dairy_items': df['has_dairy'].sum(),
                'nut_items': df['has_nuts'].sum()
            }
            
            logger.info("Menu pattern analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing menu patterns: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def export_analysis_report(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """Export analysis report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"menu_analysis_{timestamp}.json"
        
        report_file = self.processed_dir / filename
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis report saved to {report_file}")
            return str(report_file)
        except Exception as e:
            logger.error(f"Error saving analysis report: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    processor = MenuProcessor()
    
    # Create item database
    all_items_df, unique_items_df = processor.create_item_database()
    
    if not all_items_df.empty:
        # Analyze patterns
        analysis = processor.analyze_menu_patterns(all_items_df)
        
        # Print key insights
        print("\n--- Menu Analysis Report ---")
        print(f"Total menu items processed: {len(all_items_df)}")
        print(f"Unique items: {len(unique_items_df)}")
        
        print("\nMost common items:")
        for item, count in list(analysis['most_common_items'].items())[:5]:
            print(f"  {item}: {count}")
        
        print("\nItems by dining hall:")
        for hall, count in analysis['items_by_dining_hall'].items():
            print(f"  {hall}: {count}")
        
        print(f"\nAverage health score: {analysis['health_score_stats']['mean']:.2f}")
        
        # Export report
        report_file = processor.export_analysis_report(analysis)
        print(f"\nAnalysis report saved to: {report_file}")
    else:
        print("No menu data found. Please run the scraper first.")
