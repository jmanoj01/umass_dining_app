import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class UserPreferenceTracker:
    """
    Enhanced user preference tracking with analytics and insights
    """
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.data_dir = Path("user_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferences_file = self.data_dir / f"{user_id}_preferences.json"
        self.preferences = self.load_preferences()
        
        # Analytics cache
        self._analytics_cache = {}
        self._cache_timestamp = None
    
    def load_preferences(self) -> Dict[str, Any]:
        """Load existing preferences or create new"""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
                return self._create_default_preferences()
        
        return self._create_default_preferences()
    
    def _create_default_preferences(self) -> Dict[str, Any]:
        """Create default preference structure"""
        return {
            'user_id': self.user_id,
            'created_at': datetime.now().isoformat(),
            'ratings': {},  # item_id: rating (1-5)
            'history': [],   # List of items eaten
            'dietary_restrictions': [],
            'favorite_stations': [],
            'disliked_allergens': [],
            'preferences': {
                'meal_times': [],
                'cuisine_preferences': [],
                'spice_level': 'medium',
                'portion_size': 'medium'
            },
            'analytics': {
                'total_ratings': 0,
                'average_rating': 0,
                'most_rated_station': None,
                'rating_trend': []
            }
        }
    
    def save_preferences(self) -> bool:
        """Save preferences to file with error handling"""
        try:
            # Update analytics before saving
            self._update_analytics()
            
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
            return False
    
    def rate_item(self, item_id: int, rating: float, item_name: str = None, 
                  dining_hall: str = None, station: str = None) -> bool:
        """
        Rate a food item (1-5 stars) with enhanced metadata
        
        Args:
            item_id: Unique item identifier
            rating: 1-5 (1=hated it, 5=loved it)
            item_name: Optional item name for reference
            dining_hall: Dining hall where item was consumed
            station: Station where item was found
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        rating_data = {
            'rating': rating,
            'item_name': item_name,
            'dining_hall': dining_hall,
            'station': station,
            'timestamp': datetime.now().isoformat()
        }
        
        self.preferences['ratings'][str(item_id)] = rating_data
        
        # Update station preferences
        if station:
            self._update_station_preferences(station, rating)
        
        success = self.save_preferences()
        if success:
            logger.info(f"Rated {item_name or item_id}: {rating}/5")
        
        return success
    
    def add_to_history(self, item_id: int, dining_hall: str, meal_period: str, 
                      item_name: str = None, station: str = None, 
                      rating: float = None) -> bool:
        """Record that user ate this item with enhanced tracking"""
        history_entry = {
            'item_id': item_id,
            'item_name': item_name,
            'dining_hall': dining_hall,
            'meal_period': meal_period,
            'station': station,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        self.preferences['history'].append(history_entry)
        
        # Update meal time preferences
        self._update_meal_time_preferences(meal_period)
        
        success = self.save_preferences()
        if success:
            logger.info(f"Added to history: {item_name or item_id}")
        
        return success
    
    def set_dietary_restrictions(self, restrictions: List[str]) -> bool:
        """Set dietary restrictions"""
        self.preferences['dietary_restrictions'] = restrictions
        return self.save_preferences()
    
    def set_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Set general user preferences"""
        self.preferences['preferences'].update(preferences)
        return self.save_preferences()
    
    def _update_station_preferences(self, station: str, rating: float):
        """Update station preferences based on ratings"""
        if 'favorite_stations' not in self.preferences:
            self.preferences['favorite_stations'] = {}
        
        if station not in self.preferences['favorite_stations']:
            self.preferences['favorite_stations'][station] = {
                'total_ratings': 0,
                'average_rating': 0,
                'count': 0
            }
        
        station_data = self.preferences['favorite_stations'][station]
        station_data['count'] += 1
        station_data['total_ratings'] += rating
        station_data['average_rating'] = station_data['total_ratings'] / station_data['count']
    
    def _update_meal_time_preferences(self, meal_period: str):
        """Update meal time preferences"""
        if meal_period not in self.preferences['preferences']['meal_times']:
            self.preferences['preferences']['meal_times'].append(meal_period)
    
    def _update_analytics(self):
        """Update user analytics"""
        rated_df = self.get_rated_items()
        history_df = self.get_eating_history()
        
        analytics = {
            'total_ratings': len(rated_df),
            'average_rating': rated_df['rating'].mean() if len(rated_df) > 0 else 0,
            'total_meals_tracked': len(history_df),
            'most_rated_station': None,
            'rating_trend': []
        }
        
        if len(rated_df) > 0:
            # Most rated station
            if 'station' in rated_df.columns:
                station_ratings = rated_df.groupby('station')['rating'].agg(['count', 'mean'])
                if len(station_ratings) > 0:
                    analytics['most_rated_station'] = station_ratings.loc[station_ratings['count'].idxmax()].name
        
            # Rating trend (last 10 ratings)
            recent_ratings = rated_df.sort_values('timestamp').tail(10)
            analytics['rating_trend'] = recent_ratings['rating'].tolist()
        
        self.preferences['analytics'] = analytics
    
    def get_rated_items(self) -> pd.DataFrame:
        """Get all rated items as DataFrame"""
        if not self.preferences['ratings']:
            return pd.DataFrame()
        
        data = []
        for item_id, rating_data in self.preferences['ratings'].items():
            data.append({
                'item_id': int(item_id),
                'rating': rating_data['rating'],
                'item_name': rating_data.get('item_name'),
                'dining_hall': rating_data.get('dining_hall'),
                'station': rating_data.get('station'),
                'timestamp': rating_data['timestamp']
            })
        
        return pd.DataFrame(data)
    
    def get_eating_history(self) -> pd.DataFrame:
        """Get eating history as DataFrame"""
        if not self.preferences['history']:
            return pd.DataFrame()
        
        return pd.DataFrame(self.preferences['history'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive user preference statistics"""
        rated_df = self.get_rated_items()
        history_df = self.get_eating_history()
        
        stats = {
            'user_id': self.user_id,
            'total_ratings': len(rated_df),
            'average_rating': rated_df['rating'].mean() if len(rated_df) > 0 else 0,
            'total_meals_tracked': len(history_df),
            'dietary_restrictions': self.preferences['dietary_restrictions'],
            'preferences': self.preferences['preferences'],
            'analytics': self.preferences['analytics']
        }
        
        if len(rated_df) > 0:
            stats['favorite_items'] = rated_df.nlargest(5, 'rating')[['item_name', 'rating']].to_dict('records')
            stats['disliked_items'] = rated_df.nsmallest(5, 'rating')[['item_name', 'rating']].to_dict('records')
            
            # Station preferences
            if 'station' in rated_df.columns:
                station_stats = rated_df.groupby('station')['rating'].agg(['count', 'mean']).reset_index()
                stats['station_preferences'] = station_stats.to_dict('records')
            
            # Dining hall preferences
            if 'dining_hall' in rated_df.columns:
                hall_stats = rated_df.groupby('dining_hall')['rating'].agg(['count', 'mean']).reset_index()
                stats['dining_hall_preferences'] = hall_stats.to_dict('records')
        
        if len(history_df) > 0:
            # Meal period analysis
            if 'meal_period' in history_df.columns:
                meal_stats = history_df['meal_period'].value_counts().to_dict()
                stats['meal_period_frequency'] = meal_stats
            
            # Recent activity
            recent_activity = history_df.tail(10)[['item_name', 'dining_hall', 'meal_period', 'timestamp']].to_dict('records')
            stats['recent_activity'] = recent_activity
        
        return stats
    
    def get_recommendation_context(self) -> Dict[str, Any]:
        """Get context for recommendation algorithms"""
        rated_df = self.get_rated_items()
        history_df = self.get_eating_history()
        
        context = {
            'user_id': self.user_id,
            'dietary_restrictions': self.preferences.get('dietary_restrictions', []),
            'preferences': self.preferences.get('preferences', {}),
            'rated_items': set(rated_df['item_id'].tolist()) if len(rated_df) > 0 else set(),
            'favorite_stations': list(set(self.preferences.get('favorite_stations', []))),
            'meal_times': self.preferences.get('preferences', {}).get('meal_times', []),
            'average_rating': rated_df['rating'].mean() if len(rated_df) > 0 else 3.0
        }
        
        # Get item preferences by rating
        if len(rated_df) > 0:
            high_rated = rated_df[rated_df['rating'] >= 4]['item_id'].tolist()
            low_rated = rated_df[rated_df['rating'] <= 2]['item_id'].tolist()
            context['high_rated_items'] = high_rated
            context['low_rated_items'] = low_rated
        else:
            context['high_rated_items'] = []
            context['low_rated_items'] = []
        
        return context
    
    def export_data(self, format: str = 'json') -> str:
        """Export user data in specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"user_data/{self.user_id}_export_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.preferences, f, indent=2)
            return filename
        
        elif format == 'csv':
            # Export ratings and history as CSV
            rated_df = self.get_rated_items()
            history_df = self.get_eating_history()
            
            ratings_file = f"user_data/{self.user_id}_ratings_{timestamp}.csv"
            history_file = f"user_data/{self.user_id}_history_{timestamp}.csv"
            
            if len(rated_df) > 0:
                rated_df.to_csv(ratings_file, index=False)
            if len(history_df) > 0:
                history_df.to_csv(history_file, index=False)
            
            return f"Exported to {ratings_file} and {history_file}"
        
        else:
            raise ValueError("Format must be 'json' or 'csv'")
    
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Import user data from external source"""
        try:
            # Validate data structure
            required_keys = ['user_id', 'ratings', 'history']
            if not all(key in data for key in required_keys):
                raise ValueError("Invalid data structure")
            
            # Merge with existing data
            self.preferences.update(data)
            return self.save_preferences()
        
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    tracker = UserPreferenceTracker("justin_manoj")
    
    # Example: Rate some items
    tracker.rate_item(
        item_id=42, 
        rating=5, 
        item_name="Chicken Tikka Masala",
        dining_hall="worcester",
        station="International"
    )
    
    tracker.rate_item(
        item_id=108, 
        rating=2, 
        item_name="Overcooked Brussels Sprouts",
        dining_hall="franklin",
        station="Vegetarian"
    )
    
    # Set dietary preferences
    tracker.set_dietary_restrictions(['vegetarian'])
    
    # Set general preferences
    tracker.set_preferences({
        'spice_level': 'high',
        'portion_size': 'large',
        'cuisine_preferences': ['indian', 'mexican', 'italian']
    })
    
    # Add to history
    tracker.add_to_history(
        item_id=42,
        dining_hall='worcester',
        meal_period='dinner',
        item_name="Chicken Tikka Masala",
        station="International",
        rating=5
    )
    
    # View comprehensive stats
    stats = tracker.get_statistics()
    print("User Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Get recommendation context
    context = tracker.get_recommendation_context()
    print("\nRecommendation Context:")
    print(json.dumps(context, indent=2))
    
    # Export data
    export_file = tracker.export_data('json')
    print(f"\nData exported to: {export_file}")
