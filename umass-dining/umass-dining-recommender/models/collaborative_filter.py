import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class MatrixFactorization(nn.Module):
    """
    Enhanced collaborative filtering using matrix factorization
    Similar to how Netflix recommendations work, with improvements for dining data
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # User and item embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Bias terms
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize embeddings
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        
        self.global_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict rating for user-item pairs
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
        
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        
        # Apply dropout
        user_embedding = self.dropout(user_embedding)
        item_embedding = self.dropout(item_embedding)
        
        # Dot product
        dot_product = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        
        # Add biases
        user_bias = self.user_biases(user_ids)
        item_bias = self.item_biases(item_ids)
        
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        # Clamp predictions to valid rating range
        prediction = torch.clamp(prediction, 1.0, 5.0)
        
        return prediction.squeeze()
    
    def recommend(self, user_id: int, n_items: int, top_k: int = 10, 
                  exclude_rated: bool = True, rated_items: set = None) -> List[Tuple[int, float]]:
        """
        Get top K recommendations for a user
        
        Args:
            user_id: User ID
            n_items: Total number of items
            top_k: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            rated_items: Set of already rated item IDs
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        self.eval()
        with torch.no_grad():
            # Predict ratings for all items
            user_tensor = torch.LongTensor([user_id] * n_items)
            item_tensor = torch.LongTensor(range(n_items))
            
            predictions = self.forward(user_tensor, item_tensor)
            
            # Exclude already rated items if requested
            if exclude_rated and rated_items:
                mask = torch.ones(n_items, dtype=torch.bool)
                for item_id in rated_items:
                    if item_id < n_items:
                        mask[item_id] = False
                predictions = predictions[mask]
                item_tensor = item_tensor[mask]
            
            # Get top K
            top_predictions, top_indices = torch.topk(predictions, min(top_k, len(predictions)))
            
            recommendations = [
                (int(item_tensor[idx]), float(pred))
                for idx, pred in zip(top_indices, top_predictions)
            ]
        
        return recommendations

class CollaborativeFilteringTrainer:
    """
    Trainer for collaborative filtering model with enhanced features
    """
    
    def __init__(self, model: MatrixFactorization, learning_rate: float = 0.01,
                 weight_decay: float = 0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids, item_ids, ratings = batch
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs: int = 100, 
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """Train the model with early stopping"""
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

class DiningCollaborativeFilter:
    """
    Main class for dining-specific collaborative filtering
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        # Model parameters
        self.n_factors = 50
        self.learning_rate = 0.01
        self.weight_decay = 0.01
        
    def load_data(self, ratings_file: str = "user_ratings.csv") -> pd.DataFrame:
        """Load ratings data from file"""
        ratings_path = self.data_dir / ratings_file
        
        if not ratings_path.exists():
            logger.warning(f"Ratings file not found: {ratings_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(ratings_path)
            logger.info(f"Loaded {len(df)} ratings from {ratings_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load ratings data: {e}")
            return pd.DataFrame()
    
    def prepare_data(self, ratings_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        if ratings_df.empty:
            raise ValueError("No ratings data available")
        
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Map to internal IDs
        ratings_df['user_idx'] = ratings_df['user_id'].map(self.user_mapping)
        ratings_df['item_idx'] = ratings_df['item_id'].map(self.item_mapping)
        
        # Convert to tensors
        user_ids = torch.LongTensor(ratings_df['user_idx'].values)
        item_ids = torch.LongTensor(ratings_df['item_idx'].values)
        ratings = torch.FloatTensor(ratings_df['rating'].values)
        
        logger.info(f"Prepared data: {len(unique_users)} users, {len(unique_items)} items")
        
        return user_ids, item_ids, ratings
    
    def create_data_loaders(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                           ratings: torch.Tensor, batch_size: int = 64, 
                           train_split: float = 0.8) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create train and validation data loaders"""
        # Create dataset
        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
        
        # Split data
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_model(self, ratings_df: pd.DataFrame, epochs: int = 100) -> Dict[str, Any]:
        """Train the collaborative filtering model"""
        # Prepare data
        user_ids, item_ids, ratings = self.prepare_data(ratings_df)
        
        # Create model
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        self.model = MatrixFactorization(n_users, n_items, self.n_factors)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(user_ids, item_ids, ratings)
        
        # Train model
        trainer = CollaborativeFilteringTrainer(self.model, self.learning_rate, self.weight_decay)
        history = trainer.train(train_loader, val_loader, epochs)
        
        logger.info("Model training completed")
        
        return {
            'history': history,
            'n_users': n_users,
            'n_items': n_items,
            'best_val_loss': trainer.best_val_loss
        }
    
    def get_recommendations(self, user_id: int, top_k: int = 10, 
                           exclude_rated: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        
        # Get rated items if excluding
        rated_items = set()
        if exclude_rated:
            # This would need to be passed in or loaded from data
            pass
        
        # Get recommendations
        recommendations = self.model.recommend(
            user_idx, n_items, top_k, exclude_rated, rated_items
        )
        
        # Convert back to original item IDs and format
        formatted_recommendations = []
        for item_idx, predicted_rating in recommendations:
            original_item_id = self.reverse_item_mapping[item_idx]
            formatted_recommendations.append({
                'item_id': original_item_id,
                'predicted_rating': predicted_rating,
                'confidence': min(predicted_rating / 5.0, 1.0)
            })
        
        return formatted_recommendations
    
    def save_model(self, filename: str = None) -> str:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"models/collaborative_filter_{timestamp}.pkl"
        
        model_data = {
            'model_state': self.model.state_dict(),
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'n_factors': self.n_factors,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filename}")
        return filename
    
    def load_model(self, filename: str) -> bool:
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore mappings
            self.user_mapping = model_data['user_mapping']
            self.item_mapping = model_data['item_mapping']
            self.reverse_user_mapping = model_data['reverse_user_mapping']
            self.reverse_item_mapping = model_data['reverse_item_mapping']
            self.n_factors = model_data['n_factors']
            self.learning_rate = model_data['learning_rate']
            self.weight_decay = model_data['weight_decay']
            
            # Create and load model
            n_users = len(self.user_mapping)
            n_items = len(self.item_mapping)
            self.model = MatrixFactorization(n_users, n_items, self.n_factors)
            self.model.load_state_dict(model_data['model_state'])
            
            logger.info(f"Model loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate_model(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        
        # Prepare test data
        test_user_ids = torch.LongTensor([self.user_mapping[uid] for uid in test_ratings['user_id']])
        test_item_ids = torch.LongTensor([self.item_mapping[iid] for iid in test_ratings['item_id']])
        test_ratings_tensor = torch.FloatTensor(test_ratings['rating'].values)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(test_user_ids, test_item_ids)
        
        # Calculate metrics
        mse = torch.mean((predictions - test_ratings_tensor) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - test_ratings_tensor)).item()
        
        # Calculate accuracy within 1 star
        accuracy = torch.mean((torch.abs(predictions - test_ratings_tensor) <= 1.0).float()).item()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy_within_1': accuracy
        }

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_users = 100
    n_items = 200
    n_ratings = 1000
    
    sample_data = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_ratings),
        'item_id': np.random.randint(0, n_items, n_ratings),
        'rating': np.random.uniform(1, 5, n_ratings)
    })
    
    # Train model
    cf = DiningCollaborativeFilter()
    history = cf.train_model(sample_data, epochs=50)
    
    # Get recommendations
    recommendations = cf.get_recommendations(user_id=0, top_k=5)
    print("Recommendations for user 0:")
    for rec in recommendations:
        print(f"  Item {rec['item_id']}: {rec['predicted_rating']:.2f} (confidence: {rec['confidence']:.2f})")
    
    # Save model
    model_file = cf.save_model()
    print(f"Model saved to: {model_file}")
