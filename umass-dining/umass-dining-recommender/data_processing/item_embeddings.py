import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class FoodEmbeddings:
    """
    Advanced food item embedding system using sentence transformers
    Creates semantic embeddings for food items to enable similarity-based recommendations
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a pre-trained sentence transformer optimized for semantic similarity
        self.model_name = 'all-MiniLM-L6-v2'  # Fast and effective
        self.model = None
        self.embeddings = None
        self.item_mapping = []
        
        # Embedding parameters
        self.batch_size = 32
        self.max_length = 128
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_unique_items(self) -> pd.DataFrame:
        """Load the unique items database"""
        items_file = self.processed_dir / "unique_items.csv"
        
        if not items_file.exists():
            raise FileNotFoundError(f"Unique items file not found: {items_file}")
        
        try:
            df = pd.read_csv(items_file)
            logger.info(f"Loaded {len(df)} unique items")
            return df
        except Exception as e:
            logger.error(f"Failed to load unique items: {e}")
            raise
    
    def create_item_text_representation(self, row: pd.Series) -> str:
        """
        Create a rich text description for each food item
        This is what we'll embed to capture semantic meaning
        """
        parts = []
        
        # Item name (most important)
        parts.append(row['item_name'])
        
        # Description if available
        if pd.notna(row.get('description')) and row['description'].strip():
            parts.append(row['description'])
        
        # Station context
        if pd.notna(row.get('station_clean')) and row['station_clean'].strip():
            parts.append(f"from {row['station_clean']} station")
        
        # Cuisine type
        if pd.notna(row.get('cuisine_type')) and row['cuisine_type'] != 'other':
            parts.append(f"{row['cuisine_type']} cuisine")
        
        # Meal type
        if pd.notna(row.get('meal_type')) and row['meal_type'] != 'main':
            parts.append(f"for {row['meal_type']}")
        
        # Nutritional context
        if pd.notna(row.get('calories')):
            calories = row['calories']
            if calories < 200:
                parts.append("light meal")
            elif calories < 400:
                parts.append("moderate meal")
            elif calories > 600:
                parts.append("hearty meal")
            elif calories > 800:
                parts.append("high calorie meal")
        
        if pd.notna(row.get('protein')) and row['protein'] > 20:
            parts.append("high protein")
        
        if pd.notna(row.get('fat')) and row['fat'] < 10:
            parts.append("low fat")
        
        # Dietary information
        dietary_info = []
        if row.get('is_vegan', False):
            dietary_info.append("vegan")
        if row.get('is_vegetarian', False):
            dietary_info.append("vegetarian")
        if row.get('is_gluten_free', False):
            dietary_info.append("gluten-free")
        if row.get('has_dairy', False):
            dietary_info.append("contains dairy")
        if row.get('has_nuts', False):
            dietary_info.append("contains nuts")
        
        if dietary_info:
            parts.append(", ".join(dietary_info))
        
        # Health score context
        if pd.notna(row.get('health_score')):
            health_score = row['health_score']
            if health_score >= 8:
                parts.append("very healthy")
            elif health_score >= 6:
                parts.append("healthy")
            elif health_score <= 3:
                parts.append("indulgent")
        
        # Allergen information
        if pd.notna(row.get('allergens_text')) and row['allergens_text'].strip():
            allergens = row['allergens_text'].split(',')
            if allergens and allergens[0].strip():
                parts.append(f"allergens: {row['allergens_text']}")
        
        # Combine all parts
        text_representation = ". ".join(parts)
        
        # Truncate if too long
        if len(text_representation) > 500:
            text_representation = text_representation[:500] + "..."
        
        return text_representation
    
    def generate_embeddings(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate embeddings for all unique food items"""
        if self.model is None:
            self.load_model()
        
        df = self.load_unique_items()
        
        logger.info(f"Generating embeddings for {len(df)} items...")
        
        # Create text representations
        df['text_representation'] = df.apply(
            self.create_item_text_representation, 
            axis=1
        )
        
        # Generate embeddings
        texts = df['text_representation'].tolist()
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            
            self.embeddings = embeddings
            logger.info(f"Generated embeddings: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Save embeddings
        self.save_embeddings(embeddings, df)
        
        return embeddings, df
    
    def save_embeddings(self, embeddings: np.ndarray, df: pd.DataFrame):
        """Save embeddings and metadata"""
        try:
            # Save embeddings
            embeddings_file = self.embeddings_dir / "item_embeddings.npy"
            np.save(embeddings_file, embeddings)
            logger.info(f"Saved embeddings: {embeddings_file}")
            
            # Save item mapping
            item_mapping = df[['item_id', 'item_name', 'item_name_clean', 'text_representation']].to_dict('records')
            
            mapping_file = self.embeddings_dir / "item_mapping.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(item_mapping, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved item mapping: {mapping_file}")
            
            # Save model metadata
            metadata = {
                'model_name': self.model_name,
                'embedding_dimension': embeddings.shape[1],
                'num_items': len(df),
                'created_at': datetime.now().isoformat(),
                'batch_size': self.batch_size,
                'max_length': self.max_length
            }
            
            metadata_file = self.embeddings_dir / "embedding_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load pre-computed embeddings"""
        try:
            # Load embeddings
            embeddings_file = self.embeddings_dir / "item_embeddings.npy"
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
            embeddings = np.load(embeddings_file)
            self.embeddings = embeddings
            
            # Load item mapping
            mapping_file = self.embeddings_dir / "item_mapping.json"
            if not mapping_file.exists():
                raise FileNotFoundError(f"Item mapping file not found: {mapping_file}")
            
            with open(mapping_file, 'r', encoding='utf-8') as f:
                item_mapping = json.load(f)
            
            self.item_mapping = item_mapping
            
            logger.info(f"Loaded embeddings: {embeddings.shape}")
            logger.info(f"Loaded {len(item_mapping)} item mappings")
            
            return embeddings, item_mapping
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def find_similar_items(self, item_name: str, top_k: int = 10, 
                          exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Find items similar to a given item using cosine similarity
        
        Args:
            item_name: Name of the item to find similar items for
            top_k: Number of similar items to return
            exclude_self: Whether to exclude the item itself from results
        
        Returns:
            List of similar items with similarity scores
        """
        if self.embeddings is None:
            self.load_embeddings()
        
        # Find the item in the mapping
        item_found = None
        for item in self.item_mapping:
            if (item['item_name'].lower() == item_name.lower() or 
                item['item_name_clean'] == item_name.lower()):
                item_found = item
                break
        
        if not item_found:
            logger.warning(f"Item '{item_name}' not found in embeddings")
            return []
        
        item_idx = item_found['item_id']
        
        if item_idx >= len(self.embeddings):
            logger.error(f"Item index {item_idx} out of range for embeddings")
            return []
        
        # Get item embedding
        item_embedding = self.embeddings[item_idx]
        
        # Calculate cosine similarity with all items
        similarities = np.dot(self.embeddings, item_embedding)
        
        # Get top K similar items
        if exclude_self:
            # Set self-similarity to 0
            similarities[item_idx] = 0
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter out items with very low similarity
        similar_items = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > 0.1:  # Minimum similarity threshold
                similar_item = self.item_mapping[idx]
                similar_items.append({
                    'item_id': similar_item['item_id'],
                    'item_name': similar_item['item_name'],
                    'item_name_clean': similar_item['item_name_clean'],
                    'similarity': float(similarity),
                    'confidence': min(similarity, 1.0)
                })
        
        return similar_items
    
    def find_items_by_description(self, description: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find items similar to a text description
        
        Args:
            description: Text description to search for
            top_k: Number of items to return
        
        Returns:
            List of similar items
        """
        if self.model is None:
            self.load_model()
        
        if self.embeddings is None:
            self.load_embeddings()
        
        try:
            # Encode the description
            desc_embedding = self.model.encode([description], normalize_embeddings=True)[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, desc_embedding)
            
            # Get top K
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.1:
                    item = self.item_mapping[idx]
                    results.append({
                        'item_id': item['item_id'],
                        'item_name': item['item_name'],
                        'item_name_clean': item['item_name_clean'],
                        'similarity': float(similarity),
                        'confidence': min(similarity, 1.0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding items by description: {e}")
            return []
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific item"""
        if self.embeddings is None:
            self.load_embeddings()
        
        if 0 <= item_id < len(self.embeddings):
            return self.embeddings[item_id]
        
        return None
    
    def cluster_items(self, n_clusters: int = 10) -> Dict[int, List[int]]:
        """
        Cluster items based on their embeddings
        
        Args:
            n_clusters: Number of clusters to create
        
        Returns:
            Dictionary mapping cluster_id to list of item_ids
        """
        if self.embeddings is None:
            self.load_embeddings()
        
        try:
            from sklearn.cluster import KMeans
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Group items by cluster
            clusters = {}
            for item_id, cluster_id in enumerate(cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(item_id)
            
            logger.info(f"Created {len(clusters)} clusters")
            return clusters
            
        except ImportError:
            logger.error("scikit-learn not available for clustering")
            return {}
        except Exception as e:
            logger.error(f"Error clustering items: {e}")
            return {}
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embeddings"""
        if self.embeddings is None:
            self.load_embeddings()
        
        stats = {
            'num_items': len(self.embeddings),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'mean_similarity': float(np.mean(self.embeddings @ self.embeddings.T)),
            'std_similarity': float(np.std(self.embeddings @ self.embeddings.T))
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    embedder = FoodEmbeddings()
    
    try:
        # Generate embeddings
        embeddings, df = embedder.generate_embeddings()
        
        # Test similarity search
        print("\n--- Testing Similarity Search ---")
        similar = embedder.find_similar_items("chicken breast", top_k=5)
        
        if similar:
            print("\nItems similar to 'chicken breast':")
            for item in similar:
                print(f"  {item['item_name']}: {item['similarity']:.3f}")
        
        # Test description search
        print("\n--- Testing Description Search ---")
        desc_results = embedder.find_items_by_description("spicy indian curry", top_k=3)
        
        if desc_results:
            print("\nItems matching 'spicy indian curry':")
            for item in desc_results:
                print(f"  {item['item_name']}: {item['similarity']:.3f}")
        
        # Get statistics
        stats = embedder.get_embedding_statistics()
        print(f"\n--- Embedding Statistics ---")
        print(f"Number of items: {stats['num_items']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Model: {stats['model_name']}")
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        print("Please ensure you have menu data processed first by running the menu processor.")
