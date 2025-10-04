import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for dining recommendation system
    Creates comprehensive features for both items and users
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.processed_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature scalers and encoders
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed menu and user data"""
        try:
            # Load unique items
            items_file = self.processed_dir / "unique_items.csv"
            if not items_file.exists():
                raise FileNotFoundError(f"Unique items file not found: {items_file}")
            
            items_df = pd.read_csv(items_file)
            logger.info(f"Loaded {len(items_df)} unique items")
            
            # Load user ratings (if available)
            user_ratings = pd.DataFrame()
            user_data_dir = Path("user_data")
            if user_data_dir.exists():
                all_ratings = []
                for pref_file in user_data_dir.glob("*_preferences.json"):
                    try:
                        with open(pref_file, 'r') as f:
                            prefs = json.load(f)
                        
                        user_id = prefs['user_id']
                        for item_id, rating_data in prefs.get('ratings', {}).items():
                            all_ratings.append({
                                'user_id': user_id,
                                'item_id': int(item_id),
                                'rating': rating_data['rating'],
                                'timestamp': rating_data.get('timestamp', '')
                            })
                    except Exception as e:
                        logger.warning(f"Error loading user preferences from {pref_file}: {e}")
                
                if all_ratings:
                    user_ratings = pd.DataFrame(all_ratings)
                    logger.info(f"Loaded {len(user_ratings)} user ratings")
            
            return items_df, user_ratings
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_item_features(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for food items"""
        logger.info("Creating item features...")
        
        features_df = items_df.copy()
        
        # Nutritional features
        features_df = self._create_nutritional_features(features_df)
        
        # Text features
        features_df = self._create_text_features(features_df)
        
        # Categorical features
        features_df = self._create_categorical_features(features_df)
        
        # Temporal features
        features_df = self._create_temporal_features(features_df)
        
        # Interaction features
        features_df = self._create_interaction_features(features_df)
        
        # Health and dietary features
        features_df = self._create_health_features(features_df)
        
        # Popularity features
        features_df = self._create_popularity_features(features_df)
        
        logger.info(f"Created {features_df.shape[1]} features for items")
        return features_df
    
    def _create_nutritional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create nutritional-based features"""
        # Basic nutritional features
        nutritional_cols = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'sodium']
        
        for col in nutritional_cols:
            if col in df.columns:
                # Handle missing values
                df[f'{col}_filled'] = df[col].fillna(df[col].median())
                
                # Create normalized versions
                df[f'{col}_normalized'] = (df[f'{col}_filled'] - df[f'{col}_filled'].mean()) / df[f'{col}_filled'].std()
                
                # Create categorical versions
                df[f'{col}_category'] = pd.cut(df[f'{col}_filled'], 
                                             bins=5, 
                                             labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Macro ratios
        if 'calories' in df.columns and 'protein' in df.columns:
            df['protein_per_calorie'] = df['protein'] / (df['calories'] + 1)
            df['protein_density'] = df['protein'] / (df['calories'] + 1) * 100
        
        if 'fat' in df.columns and 'calories' in df.columns:
            df['fat_per_calorie'] = df['fat'] / (df['calories'] + 1)
            df['fat_density'] = df['fat'] / (df['calories'] + 1) * 100
        
        # Nutritional balance score
        nutritional_scores = []
        for _, row in df.iterrows():
            score = 0
            if pd.notna(row.get('protein')) and row['protein'] > 15:
                score += 1
            if pd.notna(row.get('fiber')) and row['fiber'] > 3:
                score += 1
            if pd.notna(row.get('fat')) and 5 <= row['fat'] <= 15:
                score += 1
            if pd.notna(row.get('sodium')) and row['sodium'] < 600:
                score += 1
            nutritional_scores.append(score)
        
        df['nutritional_balance_score'] = nutritional_scores
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        # Item name features
        df['name_length'] = df['item_name'].str.len()
        df['name_word_count'] = df['item_name'].str.split().str.len()
        df['has_numbers'] = df['item_name'].str.contains(r'\d', regex=True).astype(int)
        df['has_special_chars'] = df['item_name'].str.contains(r'[^\w\s]', regex=True).astype(int)
        
        # Description features
        if 'description' in df.columns:
            df['has_description'] = df['description'].notna().astype(int)
            df['description_length'] = df['description'].str.len().fillna(0)
            df['description_word_count'] = df['description'].str.split().str.len().fillna(0)
        
        # Cuisine complexity (based on name)
        cuisine_complexity = []
        for name in df['item_name']:
            complexity = 0
            if any(word in name.lower() for word in ['special', 'deluxe', 'premium', 'gourmet']):
                complexity += 2
            if any(word in name.lower() for word in ['spicy', 'hot', 'mild', 'sweet']):
                complexity += 1
            if any(word in name.lower() for word in ['grilled', 'roasted', 'baked', 'fried']):
                complexity += 1
            cuisine_complexity.append(complexity)
        
        df['cuisine_complexity'] = cuisine_complexity
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical features"""
        # Station features
        if 'station_clean' in df.columns:
            df['station_encoded'] = self._encode_categorical(df['station_clean'], 'station')
            
            # Station popularity
            station_counts = df['station_clean'].value_counts()
            df['station_popularity'] = df['station_clean'].map(station_counts)
        
        # Cuisine type features
        if 'cuisine_type' in df.columns:
            df['cuisine_encoded'] = self._encode_categorical(df['cuisine_type'], 'cuisine')
            
            # Cuisine diversity (how many different cuisines in the dataset)
            cuisine_counts = df['cuisine_type'].value_counts()
            df['cuisine_diversity'] = df['cuisine_type'].map(cuisine_counts)
        
        # Meal type features
        if 'meal_type' in df.columns:
            df['meal_type_encoded'] = self._encode_categorical(df['meal_type'], 'meal_type')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        # Frequency-based features
        if 'frequency' in df.columns:
            df['frequency_normalized'] = (df['frequency'] - df['frequency'].mean()) / df['frequency'].std()
            df['frequency_rank'] = df['frequency'].rank(pct=True)
            
            # Frequency categories
            df['frequency_category'] = pd.cut(df['frequency'], 
                                            bins=5, 
                                            labels=['rare', 'uncommon', 'common', 'frequent', 'very_frequent'])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different attributes"""
        # Station-cuisine interactions
        if 'station_clean' in df.columns and 'cuisine_type' in df.columns:
            df['station_cuisine_interaction'] = df['station_clean'] + '_' + df['cuisine_type']
        
        # Health-dietary interactions
        health_dietary = []
        for _, row in df.iterrows():
            score = 0
            if row.get('is_vegan', False):
                score += 3
            elif row.get('is_vegetarian', False):
                score += 2
            if row.get('is_gluten_free', False):
                score += 1
            if row.get('health_score', 0) > 7:
                score += 2
            elif row.get('health_score', 0) > 5:
                score += 1
            health_dietary.append(score)
        
        df['health_dietary_score'] = health_dietary
        
        return df
    
    def _create_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create health-related features"""
        # Dietary restriction features
        dietary_cols = ['is_vegan', 'is_vegetarian', 'is_gluten_free', 'has_dairy', 'has_nuts']
        
        for col in dietary_cols:
            if col in df.columns:
                df[f'{col}_int'] = df[col].astype(int)
        
        # Allergen count
        if 'allergens_text' in df.columns:
            df['allergen_count'] = df['allergens_text'].str.count(',') + 1
            df['allergen_count'] = df['allergen_count'].fillna(0)
        
        # Health score features
        if 'health_score' in df.columns:
            df['health_score_normalized'] = (df['health_score'] - df['health_score'].mean()) / df['health_score'].std()
            df['is_healthy'] = (df['health_score'] >= 7).astype(int)
            df['is_very_healthy'] = (df['health_score'] >= 8).astype(int)
        
        return df
    
    def _create_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create popularity-based features"""
        if 'frequency' in df.columns:
            # Popularity percentiles
            df['popularity_percentile'] = df['frequency'].rank(pct=True)
            
            # Popularity categories
            df['popularity_category'] = pd.cut(df['frequency'], 
                                             bins=4, 
                                             labels=['unpopular', 'average', 'popular', 'very_popular'])
            
            # Relative popularity within station
            if 'station_clean' in df.columns:
                station_popularity = df.groupby('station_clean')['frequency'].rank(pct=True)
                df['station_relative_popularity'] = station_popularity
        
        return df
    
    def _encode_categorical(self, series: pd.Series, name: str) -> pd.Series:
        """Encode categorical variables"""
        if name not in self.encoders:
            self.encoders[name] = LabelEncoder()
            return pd.Series(self.encoders[name].fit_transform(series.fillna('unknown')), 
                           index=series.index)
        else:
            return pd.Series(self.encoders[name].transform(series.fillna('unknown')), 
                           index=series.index)
    
    def create_user_features(self, user_ratings: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-based features"""
        if user_ratings.empty:
            logger.warning("No user ratings available for user feature creation")
            return pd.DataFrame()
        
        logger.info("Creating user features...")
        
        user_features = []
        
        for user_id in user_ratings['user_id'].unique():
            user_data = user_ratings[user_ratings['user_id'] == user_id]
            
            # Basic user stats
            features = {
                'user_id': user_id,
                'total_ratings': len(user_data),
                'average_rating': user_data['rating'].mean(),
                'rating_std': user_data['rating'].std(),
                'rating_range': user_data['rating'].max() - user_data['rating'].min()
            }
            
            # Rating distribution
            rating_counts = user_data['rating'].value_counts().sort_index()
            for rating in range(1, 6):
                features[f'rating_{rating}_count'] = rating_counts.get(rating, 0)
                features[f'rating_{rating}_proportion'] = rating_counts.get(rating, 0) / len(user_data)
            
            # Merge with item data for item-based features
            user_items = user_data.merge(items_df, on='item_id', how='left')
            
            if not user_items.empty:
                # Nutritional preferences
                nutritional_cols = ['calories', 'protein', 'carbs', 'fat']
                for col in nutritional_cols:
                    if col in user_items.columns:
                        rated_items = user_items[user_items[col].notna()]
                        if not rated_items.empty:
                            features[f'preferred_{col}'] = rated_items[col].mean()
                            features[f'{col}_variance'] = rated_items[col].std()
                
                # Station preferences
                if 'station_clean' in user_items.columns:
                    station_ratings = user_items.groupby('station_clean')['rating'].agg(['mean', 'count'])
                    features['favorite_station'] = station_ratings['mean'].idxmax() if not station_ratings.empty else 'unknown'
                    features['favorite_station_rating'] = station_ratings['mean'].max() if not station_ratings.empty else 0
                
                # Cuisine preferences
                if 'cuisine_type' in user_items.columns:
                    cuisine_ratings = user_items.groupby('cuisine_type')['rating'].agg(['mean', 'count'])
                    features['favorite_cuisine'] = cuisine_ratings['mean'].idxmax() if not cuisine_ratings.empty else 'unknown'
                    features['favorite_cuisine_rating'] = cuisine_ratings['mean'].max() if not cuisine_ratings.empty else 0
                
                # Health preferences
                if 'health_score' in user_items.columns:
                    health_ratings = user_items[user_items['health_score'].notna()]
                    if not health_ratings.empty:
                        features['preferred_health_score'] = health_ratings['health_score'].mean()
                        features['health_preference_variance'] = health_ratings['health_score'].std()
                
                # Dietary preferences
                dietary_cols = ['is_vegan', 'is_vegetarian', 'is_gluten_free']
                for col in dietary_cols:
                    if col in user_items.columns:
                        dietary_items = user_items[user_items[col] == True]
                        if not dietary_items.empty:
                            features[f'prefers_{col}'] = dietary_items['rating'].mean()
                            features[f'{col}_proportion'] = len(dietary_items) / len(user_items)
            
            user_features.append(features)
        
        user_features_df = pd.DataFrame(user_features)
        logger.info(f"Created {user_features_df.shape[1]} features for {len(user_features_df)} users")
        
        return user_features_df
    
    def create_user_item_interaction_features(self, user_ratings: pd.DataFrame, 
                                            items_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction features"""
        if user_ratings.empty:
            return pd.DataFrame()
        
        logger.info("Creating user-item interaction features...")
        
        # Merge user ratings with item features
        interactions = user_ratings.merge(items_df, on='item_id', how='left')
        
        # Create interaction features
        interaction_features = []
        
        for _, row in interactions.iterrows():
            features = {
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'timestamp': row.get('timestamp', '')
            }
            
            # User-item compatibility features
            if 'calories' in row and pd.notna(row['calories']):
                features['calories_rating_interaction'] = row['calories'] * row['rating']
            
            if 'health_score' in row and pd.notna(row['health_score']):
                features['health_rating_interaction'] = row['health_score'] * row['rating']
            
            # Station preference interaction
            if 'station_clean' in row and pd.notna(row['station_clean']):
                # This would need user's station preferences - simplified for now
                features['station_rating_interaction'] = row['rating']  # Placeholder
            
            interaction_features.append(features)
        
        interaction_df = pd.DataFrame(interaction_features)
        logger.info(f"Created {interaction_df.shape[1]} interaction features")
        
        return interaction_df
    
    def save_features(self, item_features: pd.DataFrame, user_features: pd.DataFrame = None,
                     interaction_features: pd.DataFrame = None):
        """Save engineered features"""
        try:
            # Save item features
            item_features_file = self.features_dir / "item_features.csv"
            item_features.to_csv(item_features_file, index=False)
            logger.info(f"Saved item features to {item_features_file}")
            
            # Save user features
            if user_features is not None and not user_features.empty:
                user_features_file = self.features_dir / "user_features.csv"
                user_features.to_csv(user_features_file, index=False)
                logger.info(f"Saved user features to {user_features_file}")
            
            # Save interaction features
            if interaction_features is not None and not interaction_features.empty:
                interaction_features_file = self.features_dir / "interaction_features.csv"
                interaction_features.to_csv(interaction_features_file, index=False)
                logger.info(f"Saved interaction features to {interaction_features_file}")
            
            # Save feature metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'item_features_count': len(item_features),
                'item_feature_columns': list(item_features.columns),
                'user_features_count': len(user_features) if user_features is not None else 0,
                'interaction_features_count': len(interaction_features) if interaction_features is not None else 0
            }
            
            metadata_file = self.features_dir / "feature_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved feature metadata to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise
    
    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load engineered features"""
        try:
            # Load item features
            item_features_file = self.features_dir / "item_features.csv"
            item_features = pd.read_csv(item_features_file) if item_features_file.exists() else pd.DataFrame()
            
            # Load user features
            user_features_file = self.features_dir / "user_features.csv"
            user_features = pd.read_csv(user_features_file) if user_features_file.exists() else pd.DataFrame()
            
            # Load interaction features
            interaction_features_file = self.features_dir / "interaction_features.csv"
            interaction_features = pd.read_csv(interaction_features_file) if interaction_features_file.exists() else pd.DataFrame()
            
            logger.info(f"Loaded features: {len(item_features)} items, {len(user_features)} users, {len(interaction_features)} interactions")
            
            return item_features, user_features, interaction_features
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    engineer = FeatureEngineer()
    
    try:
        # Load data
        items_df, user_ratings = engineer.load_data()
        
        if items_df.empty:
            print("No item data found. Please run the menu processor first.")
            exit(1)
        
        # Create item features
        item_features = engineer.create_item_features(items_df)
        
        # Create user features
        user_features = engineer.create_user_features(user_ratings, items_df)
        
        # Create interaction features
        interaction_features = engineer.create_user_item_interaction_features(user_ratings, items_df)
        
        # Save features
        engineer.save_features(item_features, user_features, interaction_features)
        
        print(f"\nFeature engineering completed!")
        print(f"Item features: {item_features.shape}")
        print(f"User features: {user_features.shape}")
        print(f"Interaction features: {interaction_features.shape}")
        
        # Show some example features
        print(f"\nSample item features:")
        print(item_features[['item_name', 'calories', 'health_score', 'cuisine_complexity']].head())
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        print("Please ensure you have processed menu data first.")
