import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    Content-based recommendation system for dining items
    Uses item features and descriptions to find similar items
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.item_features = None
        self.item_similarity_matrix = None
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.vectorizer = None
        self.scaler = StandardScaler()
        
    def load_item_data(self, items_file: str = "unique_items.csv") -> pd.DataFrame:
        """Load item data from file"""
        items_path = self.data_dir / items_file
        
        if not items_path.exists():
            logger.warning(f"Items file not found: {items_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(items_path)
            logger.info(f"Loaded {len(df)} items from {items_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load items data: {e}")
            return pd.DataFrame()
    
    def create_item_features(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for items"""
        if items_df.empty:
            raise ValueError("No items data available")
        
        # Create item mapping
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(items_df['item_id'])}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        features = []
        
        for _, item in items_df.iterrows():
            feature_vector = self._extract_item_features(item)
            features.append(feature_vector)
        
        feature_df = pd.DataFrame(features, index=items_df['item_id'])
        self.item_features = feature_df
        
        logger.info(f"Created feature matrix: {feature_df.shape}")
        return feature_df
    
    def _extract_item_features(self, item: pd.Series) -> Dict[str, Any]:
        """Extract features from a single item"""
        features = {}
        
        # Text features
        name = str(item.get('item_name', '')).lower()
        description = str(item.get('description', '')).lower()
        station = str(item.get('station', '')).lower()
        
        # Combine text for TF-IDF
        text_content = f"{name} {description} {station}"
        features['text_content'] = text_content
        
        # Nutritional features
        features['calories'] = float(item.get('calories', 0)) if pd.notna(item.get('calories')) else 0
        features['protein'] = float(item.get('protein', 0)) if pd.notna(item.get('protein')) else 0
        features['carbs'] = float(item.get('carbs', 0)) if pd.notna(item.get('carbs')) else 0
        features['fat'] = float(item.get('fat', 0)) if pd.notna(item.get('fat')) else 0
        
        # Dietary features
        allergens = str(item.get('allergens', '')).lower()
        features['is_vegan'] = 1 if 'vegan' in allergens else 0
        features['is_vegetarian'] = 1 if 'vegetarian' in allergens else 0
        features['is_gluten_free'] = 1 if 'gluten' not in allergens else 0
        features['has_dairy'] = 1 if 'dairy' in allergens else 0
        features['has_nuts'] = 1 if 'nut' in allergens else 0
        
        # Station features (one-hot encoding)
        station_mapping = {
            'international': 'international',
            'grill': 'grill',
            'pizza': 'pizza',
            'salad': 'salad',
            'vegetarian': 'vegetarian',
            'dessert': 'dessert',
            'soup': 'soup'
        }
        
        for station_key, station_name in station_mapping.items():
            features[f'station_{station_name}'] = 1 if station_key in station else 0
        
        # Meal type features
        meal_indicators = {
            'breakfast': ['egg', 'pancake', 'waffle', 'cereal', 'toast', 'bacon'],
            'lunch': ['sandwich', 'wrap', 'salad', 'soup'],
            'dinner': ['pasta', 'rice', 'meat', 'chicken', 'beef', 'fish'],
            'dessert': ['cake', 'pie', 'ice cream', 'cookie', 'brownie']
        }
        
        for meal_type, indicators in meal_indicators.items():
            features[f'meal_{meal_type}'] = 1 if any(indicator in name for indicator in indicators) else 0
        
        # Cuisine features
        cuisine_indicators = {
            'italian': ['pasta', 'pizza', 'marinara', 'parmesan', 'mozzarella'],
            'mexican': ['taco', 'burrito', 'salsa', 'guacamole', 'jalapeno'],
            'asian': ['rice', 'noodle', 'soy', 'ginger', 'sesame', 'teriyaki'],
            'indian': ['curry', 'tikka', 'masala', 'naan', 'dal'],
            'american': ['burger', 'fries', 'cheese', 'bacon', 'grilled']
        }
        
        for cuisine, indicators in cuisine_indicators.items():
            features[f'cuisine_{cuisine}'] = 1 if any(indicator in name for indicator in indicators) else 0
        
        # Healthiness score
        health_score = 0
        if features['calories'] > 0:
            if features['calories'] < 200:
                health_score += 2
            elif features['calories'] < 400:
                health_score += 1
            
            if features['protein'] > 15:
                health_score += 1
            if features['fat'] < 10:
                health_score += 1
            if features['is_vegetarian']:
                health_score += 1
        
        features['health_score'] = health_score
        
        return features
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute item similarity matrix using multiple similarity measures"""
        if self.item_features is None:
            raise ValueError("Item features not computed. Call create_item_features() first.")
        
        # Separate text and numerical features
        text_features = self.item_features['text_content'].tolist()
        numerical_features = self.item_features.drop('text_content', axis=1)
        
        # TF-IDF for text features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        text_matrix = self.vectorizer.fit_transform(text_features).toarray()
        
        # Normalize numerical features
        numerical_matrix = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        combined_features = np.hstack([text_matrix, numerical_matrix])
        
        # Compute cosine similarity
        self.item_similarity_matrix = cosine_similarity(combined_features)
        
        logger.info(f"Computed similarity matrix: {self.item_similarity_matrix.shape}")
        return self.item_similarity_matrix
    
    def get_similar_items(self, item_id: int, top_k: int = 10, 
                         exclude_self: bool = True) -> List[Dict[str, Any]]:
        """Get items similar to a given item"""
        if self.item_similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
        
        if item_id not in self.item_mapping:
            logger.warning(f"Item {item_id} not found in data")
            return []
        
        item_idx = self.item_mapping[item_id]
        similarities = self.item_similarity_matrix[item_idx]
        
        # Get top similar items
        if exclude_self:
            # Set self-similarity to 0
            similarities[item_idx] = 0
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_items = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include items with positive similarity
                original_item_id = self.reverse_item_mapping[idx]
                similar_items.append({
                    'item_id': original_item_id,
                    'similarity': float(similarities[idx]),
                    'confidence': min(similarities[idx], 1.0)
                })
        
        return similar_items
    
    def recommend_for_user(self, user_preferences: Dict[str, Any], 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """Get content-based recommendations for a user based on their preferences"""
        if self.item_similarity_matrix is None:
            raise ValueError("Similarity matrix not computed")
        
        # Get user's liked items
        liked_items = user_preferences.get('liked_items', set())
        disliked_items = user_preferences.get('disliked_items', set())
        
        if not liked_items:
            logger.warning("No liked items found for user")
            return []
        
        # Calculate recommendation scores
        item_scores = {}
        
        for liked_item_id in liked_items:
            if liked_item_id in self.item_mapping:
                similar_items = self.get_similar_items(liked_item_id, top_k=50)
                
                for similar_item in similar_items:
                    item_id = similar_item['item_id']
                    similarity = similar_item['similarity']
                    
                    if item_id not in disliked_items:
                        if item_id not in item_scores:
                            item_scores[item_id] = 0
                        item_scores[item_id] += similarity
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:top_k]:
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'confidence': min(score / len(liked_items), 1.0)
            })
        
        return recommendations
    
    def get_item_profile(self, item_id: int) -> Dict[str, Any]:
        """Get detailed profile of an item"""
        if item_id not in self.item_mapping:
            return {}
        
        item_idx = self.item_mapping[item_id]
        features = self.item_features.iloc[item_idx]
        
        profile = {
            'item_id': item_id,
            'features': features.to_dict(),
            'nutritional_info': {
                'calories': features.get('calories', 0),
                'protein': features.get('protein', 0),
                'carbs': features.get('carbs', 0),
                'fat': features.get('fat', 0)
            },
            'dietary_info': {
                'is_vegan': bool(features.get('is_vegan', 0)),
                'is_vegetarian': bool(features.get('is_vegetarian', 0)),
                'is_gluten_free': bool(features.get('is_gluten_free', 0)),
                'has_dairy': bool(features.get('has_dairy', 0)),
                'has_nuts': bool(features.get('has_nuts', 0))
            },
            'health_score': features.get('health_score', 0)
        }
        
        return profile
    
    def find_items_by_criteria(self, criteria: Dict[str, Any], 
                              top_k: int = 20) -> List[Dict[str, Any]]:
        """Find items matching specific criteria"""
        if self.item_features is None:
            raise ValueError("Item features not computed")
        
        # Filter items based on criteria
        mask = pd.Series([True] * len(self.item_features), index=self.item_features.index)
        
        for key, value in criteria.items():
            if key in self.item_features.columns:
                if isinstance(value, bool):
                    mask &= (self.item_features[key] == (1 if value else 0))
                elif isinstance(value, (int, float)):
                    mask &= (self.item_features[key] == value)
                elif isinstance(value, dict):
                    if 'min' in value:
                        mask &= (self.item_features[key] >= value['min'])
                    if 'max' in value:
                        mask &= (self.item_features[key] <= value['max'])
        
        # Get matching items
        matching_items = self.item_features[mask]
        
        # Sort by health score or other criteria
        if 'health_score' in matching_items.columns:
            matching_items = matching_items.sort_values('health_score', ascending=False)
        
        results = []
        for item_id, features in matching_items.head(top_k).iterrows():
            results.append({
                'item_id': item_id,
                'health_score': features.get('health_score', 0),
                'calories': features.get('calories', 0),
                'protein': features.get('protein', 0)
            })
        
        return results
    
    def save_model(self, filename: str = None) -> str:
        """Save the trained model"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"models/content_based_{timestamp}.pkl"
        
        model_data = {
            'item_features': self.item_features,
            'item_similarity_matrix': self.item_similarity_matrix,
            'item_mapping': self.item_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filename}")
        return filename
    
    def load_model(self, filename: str) -> bool:
        """Load a trained model"""
        try:
            import pickle
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.item_features = model_data['item_features']
            self.item_similarity_matrix = model_data['item_similarity_matrix']
            self.item_mapping = model_data['item_mapping']
            self.reverse_item_mapping = model_data['reverse_item_mapping']
            self.vectorizer = model_data['vectorizer']
            self.scaler = model_data['scaler']
            
            logger.info(f"Model loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_items = pd.DataFrame({
        'item_id': [1, 2, 3, 4, 5],
        'item_name': ['Chicken Tikka Masala', 'Caesar Salad', 'Pizza Slice', 'Grilled Salmon', 'Veggie Burger'],
        'description': ['Spicy Indian curry', 'Fresh romaine lettuce', 'Cheese pizza', 'Grilled fish', 'Plant-based burger'],
        'station': ['International', 'Salad Bar', 'Pizza', 'Grill', 'Vegetarian'],
        'calories': [350, 150, 300, 250, 200],
        'protein': [25, 8, 15, 30, 18],
        'carbs': [20, 10, 35, 5, 25],
        'fat': [15, 8, 12, 10, 8],
        'allergens': ['dairy', 'gluten', 'dairy,gluten', '', 'gluten']
    })
    
    # Train model
    cb = ContentBasedRecommender()
    features = cb.create_item_features(sample_items)
    similarity_matrix = cb.compute_similarity_matrix()
    
    # Get similar items
    similar = cb.get_similar_items(item_id=1, top_k=3)
    print("Items similar to Chicken Tikka Masala:")
    for item in similar:
        print(f"  Item {item['item_id']}: {item['similarity']:.3f}")
    
    # Get recommendations for user
    user_prefs = {
        'liked_items': {1, 4},  # User likes Chicken Tikka Masala and Grilled Salmon
        'disliked_items': {2}   # User doesn't like Caesar Salad
    }
    
    recommendations = cb.recommend_for_user(user_prefs, top_k=3)
    print("\nRecommendations for user:")
    for rec in recommendations:
        print(f"  Item {rec['item_id']}: {rec['score']:.3f}")
    
    # Save model
    model_file = cb.save_model()
    print(f"Model saved to: {model_file}")
