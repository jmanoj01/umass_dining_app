import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Advanced hybrid recommendation system that combines:
    1. Collaborative Filtering (user behavior patterns)
    2. Content-Based (item similarity)
    3. Contextual (time, weather, dining hall)
    4. Popularity (fallback recommendations)
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_dir = Path("data/processed")
        self.embeddings_dir = Path("data/embeddings")
        
        # Load necessary data and models
        self.load_data()
        
        # Recommendation weights
        self.weights = {
            'collaborative': 0.4,
            'content_based': 0.3,
            'contextual': 0.2,
            'popularity': 0.1
        }
    
    def load_data(self):
        """Load all necessary models and data"""
        try:
            # Load item embeddings
            embeddings_file = self.embeddings_dir / "item_embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            else:
                logger.warning("Item embeddings not found")
                self.embeddings = None
            
            # Load item mapping
            mapping_file = self.embeddings_dir / "item_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.item_mapping = json.load(f)
            else:
                self.item_mapping = {}
            
            # Load unique items
            items_file = self.data_dir / "unique_items.csv"
            if items_file.exists():
                self.items_df = pd.read_csv(items_file)
                logger.info(f"Loaded {len(self.items_df)} unique items")
            else:
                logger.warning("Unique items file not found")
                self.items_df = pd.DataFrame()
            
            # Load user preferences
            from models.user_preferences import UserPreferenceTracker
            self.user_tracker = UserPreferenceTracker(self.user_id)
            
            # Load collaborative filtering model
            self.cf_model = None
            self.user_to_idx = {}
            self.item_to_idx = {}
            self.load_collaborative_model()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_collaborative_model(self):
        """Load collaborative filtering model if available"""
        try:
            from models.collaborative_filter import DiningCollaborativeFilter
            cf = DiningCollaborativeFilter()
            
            # Look for saved model
            model_files = list(Path("models").glob("collaborative_filter_*.pkl"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                if cf.load_model(str(latest_model)):
                    self.cf_model = cf.model
                    self.user_to_idx = cf.user_mapping
                    self.item_to_idx = cf.item_mapping
                    logger.info("Collaborative filtering model loaded")
                else:
                    logger.warning("Failed to load collaborative model")
        except Exception as e:
            logger.warning(f"Could not load collaborative model: {e}")
    
    def collaborative_recommendations(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations from collaborative filtering"""
        if self.cf_model is None:
            return []
        
        if self.user_id not in self.user_to_idx:
            logger.warning(f"User {self.user_id} not found in collaborative model")
            return []
        
        try:
            user_idx = self.user_to_idx[self.user_id]
            recommendations = self.cf_model.recommend(
                user_idx, 
                len(self.item_to_idx), 
                top_k=top_k
            )
            
            # Convert back to item IDs
            idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
            
            return [
                {
                    'item_id': idx_to_item[item_idx], 
                    'score': score, 
                    'method': 'collaborative',
                    'confidence': min(score / 5.0, 1.0)
                }
                for item_idx, score in recommendations
            ]
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def content_based_recommendations(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """Recommend items similar to highly rated items"""
        if self.embeddings is None:
            return []
        
        rated_items = self.user_tracker.get_rated_items()
        
        if len(rated_items) == 0:
            return []
        
        # Get highly rated items (4+ stars)
        liked_items = rated_items[rated_items['rating'] >= 4]
        
        if len(liked_items) == 0:
            # Fallback to all rated items
            liked_items = rated_items[rated_items['rating'] >= 3]
        
        if len(liked_items) == 0:
            return []
        
        try:
            # Get embeddings of liked items
            liked_item_ids = liked_items['item_id'].values
            valid_ids = [id for id in liked_item_ids if id < len(self.embeddings)]
            
            if not valid_ids:
                return []
            
            liked_embeddings = self.embeddings[valid_ids]
            
            # Weight by rating
            ratings = liked_items[liked_items['item_id'].isin(valid_ids)]['rating'].values
            weights = ratings / ratings.sum()
            
            # Weighted average embedding of liked items
            user_preference_vector = np.average(liked_embeddings, axis=0, weights=weights)
            
            # Calculate similarity with all items
            similarities = np.dot(self.embeddings, user_preference_vector) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(user_preference_vector) + 1e-8
            )
            
            # Get top K
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter out already rated items
            rated_item_ids = set(rated_items['item_id'].values)
            
            recommendations = []
            for idx in top_indices:
                if idx not in rated_item_ids and similarities[idx] > 0.1:
                    recommendations.append({
                        'item_id': int(idx),
                        'score': float(similarities[idx]),
                        'method': 'content_based',
                        'confidence': min(similarities[idx], 1.0)
                    })
            
            return recommendations[:top_k]
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def contextual_boost(self, recommendations: List[Dict[str, Any]], 
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Boost recommendations based on context
        
        Args:
            recommendations: List of recommendation dicts
            context: Dict with 'meal_period', 'weather', 'day_of_week', etc.
        """
        if not recommendations:
            return recommendations
        
        try:
            for rec in recommendations:
                item_id = rec['item_id']
                
                # Get item data
                item_data = self.items_df[self.items_df['item_id'] == item_id]
                if len(item_data) == 0:
                    continue
                
                item_data = item_data.iloc[0]
                item_name = item_data.get('item_name', '').lower()
                
                # Boost based on meal period appropriateness
                meal_period = context.get('meal_period', '')
                if meal_period == 'breakfast':
                    breakfast_keywords = ['eggs', 'pancake', 'waffle', 'bacon', 'cereal', 'toast', 'bagel']
                    if any(keyword in item_name for keyword in breakfast_keywords):
                        rec['score'] *= 1.3
                        rec['context_boost'] = 'breakfast_appropriate'
                
                elif meal_period == 'lunch':
                    lunch_keywords = ['sandwich', 'wrap', 'salad', 'soup', 'pizza']
                    if any(keyword in item_name for keyword in lunch_keywords):
                        rec['score'] *= 1.2
                        rec['context_boost'] = 'lunch_appropriate'
                
                elif meal_period == 'dinner':
                    dinner_keywords = ['pasta', 'rice', 'meat', 'chicken', 'beef', 'fish', 'curry']
                    if any(keyword in item_name for keyword in dinner_keywords):
                        rec['score'] *= 1.2
                        rec['context_boost'] = 'dinner_appropriate'
                
                # Weather-based boost
                weather = context.get('weather', '')
                if weather == 'cold':
                    cold_keywords = ['soup', 'stew', 'hot', 'warm', 'chili']
                    if any(keyword in item_name for keyword in cold_keywords):
                        rec['score'] *= 1.2
                        rec['context_boost'] = 'weather_appropriate'
                elif weather == 'hot':
                    hot_keywords = ['salad', 'cold', 'fresh', 'fruit', 'ice']
                    if any(keyword in item_name for keyword in hot_keywords):
                        rec['score'] *= 1.2
                        rec['context_boost'] = 'weather_appropriate'
                
                # Day of week preferences
                day_of_week = context.get('day_of_week', '')
                if day_of_week in ['Friday', 'Saturday']:
                    # Weekend preferences
                    if any(keyword in item_name for keyword in ['pizza', 'burger', 'fries']):
                        rec['score'] *= 1.1
                
                # Dietary restrictions (hard filter)
                dietary_restrictions = self.user_tracker.preferences.get('dietary_restrictions', [])
                
                if 'vegan' in dietary_restrictions:
                    if not item_data.get('is_vegan', False):
                        rec['score'] *= 0.1  # Heavy penalty
                        rec['dietary_filter'] = 'not_vegan'
                
                if 'vegetarian' in dietary_restrictions:
                    if not item_data.get('is_vegetarian', False):
                        rec['score'] *= 0.1
                        rec['dietary_filter'] = 'not_vegetarian'
                
                # Allergy check (hard filter)
                disliked_allergens = self.user_tracker.preferences.get('disliked_allergens', [])
                item_allergens = str(item_data.get('allergens', '')).lower()
                
                for allergen in disliked_allergens:
                    if allergen.lower() in item_allergens:
                        rec['score'] = 0  # Eliminate completely
                        rec['allergy_filter'] = f'contains_{allergen}'
                        break
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in contextual boost: {e}")
            return recommendations
    
    def popularity_recommendations(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Fallback: recommend most popular items"""
        if self.items_df.empty:
            return []
        
        try:
            # Sort by frequency
            popular_items = self.items_df.nlargest(top_k, 'frequency')
            
            max_freq = self.items_df['frequency'].max()
            
            return [
                {
                    'item_id': int(row['item_id']),
                    'score': float(row['frequency']) / max_freq if max_freq > 0 else 0.5,
                    'method': 'popularity',
                    'confidence': 0.5
                }
                for _, row in popular_items.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error in popularity recommendations: {e}")
            return []
    
    def get_recommendations(self, dining_hall: str = None, meal_period: str = None, 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations
        
        Args:
            dining_hall: Specific dining hall to recommend from
            meal_period: 'breakfast', 'lunch', 'dinner'
            top_k: Number of recommendations
        
        Returns:
            List of recommended items with metadata
        """
        try:
            all_recommendations = []
            
            # Get recommendations from different methods
            cf_recs = self.collaborative_recommendations(top_k=20)
            content_recs = self.content_based_recommendations(top_k=20)
            popular_recs = self.popularity_recommendations(top_k=10)
            
            # Combine with weights
            all_recommendations.extend([
                {**r, 'score': r['score'] * self.weights['collaborative']} 
                for r in cf_recs
            ])
            all_recommendations.extend([
                {**r, 'score': r['score'] * self.weights['content_based']} 
                for r in content_recs
            ])
            all_recommendations.extend([
                {**r, 'score': r['score'] * self.weights['popularity']} 
                for r in popular_recs
            ])
            
            # Apply contextual boosts
            context = {
                'meal_period': meal_period or self._infer_meal_period(),
                'weather': self._get_current_weather(),
                'day_of_week': datetime.now().strftime('%A')
            }
            
            all_recommendations = self.contextual_boost(all_recommendations, context)
            
            # Remove duplicates (keep highest score)
            seen = {}
            for rec in all_recommendations:
                item_id = rec['item_id']
                if item_id not in seen or rec['score'] > seen[item_id]['score']:
                    seen[item_id] = rec
            
            all_recommendations = list(seen.values())
            
            # Filter by dining hall if specified
            if dining_hall:
                all_recommendations = self._filter_by_dining_hall(
                    all_recommendations, 
                    dining_hall, 
                    meal_period or context['meal_period']
                )
            
            # Sort by score and get top K
            all_recommendations.sort(key=lambda x: x['score'], reverse=True)
            top_recommendations = all_recommendations[:top_k]
            
            # Enrich with item details
            return self._enrich_recommendations(top_recommendations)
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _infer_meal_period(self) -> str:
        """Infer current meal period from time"""
        hour = datetime.now().hour
        
        if 7 <= hour < 11:
            return 'breakfast'
        elif 11 <= hour < 16:
            return 'lunch'
        else:
            return 'dinner'
    
    def _get_current_weather(self) -> str:
        """Get current weather (simplified)"""
        # In production, call weather API
        # For now, simple heuristic based on season
        month = datetime.now().month
        
        if month in [12, 1, 2]:  # Winter
            return 'cold'
        elif month in [6, 7, 8]:  # Summer
            return 'hot'
        else:
            return 'moderate'
    
    def _filter_by_dining_hall(self, recommendations: List[Dict[str, Any]], 
                              dining_hall: str, meal_period: str) -> List[Dict[str, Any]]:
        """Filter recommendations to only show items available at specific dining hall"""
        try:
            # Load today's menu
            today = datetime.now().strftime('%Y%m%d')
            menu_file = Path(f"data/raw/menus/menu_{today}.json")
            
            if not menu_file.exists():
                logger.warning("Menu file not found, cannot filter by dining hall")
                return recommendations
            
            with open(menu_file, 'r') as f:
                menu_data = json.load(f)
            
            # Get available items
            available_items = set()
            
            if dining_hall in menu_data.get('menus', {}):
                hall_menu = menu_data['menus'][dining_hall]
                if meal_period in hall_menu.get('meals', {}):
                    for station, items in hall_menu['meals'][meal_period].items():
                        for item in items:
                            # Match by cleaned name
                            item_clean = item['name'].lower().strip()
                            matching_items = self.items_df[
                                self.items_df['item_name_clean'] == item_clean
                            ]
                            if len(matching_items) > 0:
                                available_items.add(int(matching_items.iloc[0]['item_id']))
            
            # Filter recommendations
            filtered = [r for r in recommendations if r['item_id'] in available_items]
            
            if not filtered and recommendations:
                logger.warning(f"No items available at {dining_hall} for {meal_period}")
                return recommendations[:5]  # Return top 5 anyway
            
            return filtered
        except Exception as e:
            logger.error(f"Error filtering by dining hall: {e}")
            return recommendations
    
    def _enrich_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add full item details to recommendations"""
        enriched = []
        
        for rec in recommendations:
            item_id = rec['item_id']
            item_data = self.items_df[self.items_df['item_id'] == item_id]
            
            if len(item_data) == 0:
                continue
            
            item_data = item_data.iloc[0]
            
            enriched.append({
                'item_id': item_id,
                'item_name': item_data.get('item_name', 'Unknown'),
                'score': rec['score'],
                'method': rec['method'],
                'confidence': rec.get('confidence', 0.5),
                'station': item_data.get('station', ''),
                'calories': item_data.get('calories'),
                'protein': item_data.get('protein'),
                'carbs': item_data.get('carbs'),
                'fat': item_data.get('fat'),
                'allergens': item_data.get('allergens', ''),
                'is_vegan': item_data.get('is_vegan', False),
                'is_vegetarian': item_data.get('is_vegetarian', False),
                'frequency': item_data.get('frequency', 0),
                'context_boost': rec.get('context_boost'),
                'dietary_filter': rec.get('dietary_filter'),
                'allergy_filter': rec.get('allergy_filter')
            })
        
        return enriched
    
    def explain_recommendation(self, item_id: int) -> List[str]:
        """Explain why an item was recommended"""
        explanations = []
        
        try:
            # Check if similar to liked items
            rated_items = self.user_tracker.get_rated_items()
            liked_items = rated_items[rated_items['rating'] >= 4]
            
            if len(liked_items) > 0 and self.embeddings is not None:
                # Find most similar liked item
                liked_ids = liked_items['item_id'].values
                valid_liked_ids = [id for id in liked_ids if id < len(self.embeddings)]
                
                if valid_liked_ids and item_id < len(self.embeddings):
                    item_embedding = self.embeddings[item_id]
                    liked_embeddings = self.embeddings[valid_liked_ids]
                    
                    similarities = np.dot(liked_embeddings, item_embedding) / (
                        np.linalg.norm(liked_embeddings, axis=1) * np.linalg.norm(item_embedding) + 1e-8
                    )
                    
                    max_sim_idx = similarities.argmax()
                    max_similarity = similarities[max_sim_idx]
                    
                    if max_similarity > 0.7:
                        similar_item = liked_items.iloc[max_sim_idx]
                        explanations.append(
                            f"Similar to '{similar_item['item_name']}' which you rated {similar_item['rating']}/5"
                        )
            
            # Check nutritional preferences
            item_data = self.items_df[self.items_df['item_id'] == item_id]
            if len(item_data) > 0:
                item_data = item_data.iloc[0]
                
                # Check calorie preferences
                if not rated_items.empty:
                    rated_with_nutrition = rated_items.merge(
                        self.items_df, on='item_id', how='left'
                    )
                    avg_rated_calories = rated_with_nutrition['calories'].mean()
                    
                    if pd.notna(item_data.get('calories')) and pd.notna(avg_rated_calories):
                        if abs(item_data['calories'] - avg_rated_calories) < 100:
                            explanations.append(
                                f"Matches your typical calorie range (~{int(avg_rated_calories)} cal)"
                            )
                
                # Check dietary match
                dietary_restrictions = self.user_tracker.preferences.get('dietary_restrictions', [])
                
                if 'vegan' in dietary_restrictions and item_data.get('is_vegan'):
                    explanations.append("Matches your vegan diet")
                elif 'vegetarian' in dietary_restrictions and item_data.get('is_vegetarian'):
                    explanations.append("Matches your vegetarian diet")
                
                # Check station preferences
                favorite_stations = self.user_tracker.preferences.get('favorite_stations', [])
                if item_data.get('station') in favorite_stations:
                    explanations.append(f"From your favorite station: {item_data['station']}")
            
            # Check popularity
            if item_data.get('frequency', 0) > self.items_df['frequency'].quantile(0.8):
                explanations.append("Popular among UMass students")
            
            if not explanations:
                explanations.append("Recommended based on your preferences")
            
            return explanations
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return ["Recommended based on your preferences"]
    
    def get_recommendation_insights(self) -> Dict[str, Any]:
        """Get insights about the recommendation process"""
        try:
            rated_items = self.user_tracker.get_rated_items()
            
            insights = {
                'user_id': self.user_id,
                'total_ratings': len(rated_items),
                'average_rating': rated_items['rating'].mean() if len(rated_items) > 0 else 0,
                'dietary_restrictions': self.user_tracker.preferences.get('dietary_restrictions', []),
                'favorite_stations': list(self.user_tracker.preferences.get('favorite_stations', {}).keys()),
                'model_availability': {
                    'collaborative': self.cf_model is not None,
                    'content_based': self.embeddings is not None,
                    'item_database': not self.items_df.empty
                },
                'recommendation_weights': self.weights
            }
            
            return insights
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    recommender = HybridRecommender("justin_manoj")
    
    # Get recommendations for lunch at Worcester
    recommendations = recommender.get_recommendations(
        dining_hall='worcester',
        meal_period='lunch',
        top_k=10
    )
    
    print("\n🍽️ Top Recommendations for Lunch at Worcester:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['item_name']}")
        print(f"   Score: {rec['score']:.2f} | Method: {rec['method']} | Confidence: {rec['confidence']:.2f}")
        print(f"   Station: {rec['station']} | Calories: {rec.get('calories', 'N/A')}")
        
        # Get explanation
        explanations = recommender.explain_recommendation(rec['item_id'])
        print(f"   Why: {' | '.join(explanations)}")
        print()
    
    # Get insights
    insights = recommender.get_recommendation_insights()
    print("\n📊 Recommendation Insights:")
    print(f"Total ratings: {insights['total_ratings']}")
    print(f"Average rating: {insights['average_rating']:.2f}")
    print(f"Model availability: {insights['model_availability']}")
