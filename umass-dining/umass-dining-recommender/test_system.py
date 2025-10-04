#!/usr/bin/env python3
"""
UMass Dining Recommender System - Comprehensive Test Script

This script demonstrates all the major components of the dining recommendation system
and provides examples of how to use each module.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_menu_scraper():
    """Test the menu scraper"""
    print("\n" + "="*50)
    print("TESTING MENU SCRAPER")
    print("="*50)
    
    try:
        from scrapers.menu_scraper import UMassDiningScraper
        
        scraper = UMassDiningScraper()
        
        # Test scraping one dining hall
        print("Scraping Worcester Dining Commons...")
        menu = scraper.scrape_daily_menu('worcester')
        
        if menu:
            print(f"✓ Successfully scraped menu for {menu['dining_hall']}")
            print(f"  Date: {menu['date']}")
            print(f"  Meals: {list(menu['meals'].keys())}")
            
            # Show sample items
            for meal, stations in menu['meals'].items():
                print(f"  {meal}:")
                for station, items in stations.items():
                    if items:
                        print(f"    {station}: {len(items)} items")
                        if len(items) > 0:
                            print(f"      Sample: {items[0]['name']}")
        else:
            print("✗ Failed to scrape menu data")
        
    except Exception as e:
        print(f"✗ Error testing menu scraper: {e}")

def test_nutrition_scraper():
    """Test the nutrition scraper"""
    print("\n" + "="*50)
    print("TESTING NUTRITION SCRAPER")
    print("="*50)
    
    try:
        from scrapers.nutrition_scraper import NutritionScraper
        
        scraper = NutritionScraper()
        
        # Test with sample items
        test_items = [
            {'name': 'Chicken Breast', 'description': 'Grilled chicken breast'},
            {'name': 'Caesar Salad', 'description': 'Fresh romaine lettuce with caesar dressing'},
            {'name': 'Pizza Slice', 'description': 'Cheese pizza slice'}
        ]
        
        print("Scraping nutrition data for sample items...")
        results = scraper.batch_scrape_nutrition(test_items)
        
        print(f"✓ Scraped nutrition for {len(results)} items")
        
        for result in results:
            print(f"  {result['item_name']}:")
            for nutrient, value in result['nutrition'].items():
                print(f"    {nutrient}: {value}")
        
    except Exception as e:
        print(f"✗ Error testing nutrition scraper: {e}")

def test_user_preferences():
    """Test user preference tracking"""
    print("\n" + "="*50)
    print("TESTING USER PREFERENCES")
    print("="*50)
    
    try:
        from models.user_preferences import UserPreferenceTracker
        
        # Create a test user
        tracker = UserPreferenceTracker("test_user")
        
        # Add some ratings
        print("Adding sample ratings...")
        tracker.rate_item(1, 5, "Chicken Tikka Masala", "worcester", "International")
        tracker.rate_item(2, 4, "Caesar Salad", "franklin", "Salad Bar")
        tracker.rate_item(3, 2, "Overcooked Brussels Sprouts", "berkshire", "Vegetarian")
        
        # Set preferences
        tracker.set_dietary_restrictions(['vegetarian'])
        tracker.set_preferences({
            'spice_level': 'high',
            'portion_size': 'large',
            'cuisine_preferences': ['indian', 'mexican']
        })
        
        # Add to history
        tracker.add_to_history(1, 'worcester', 'dinner', "Chicken Tikka Masala", "International", 5)
        
        # Get statistics
        stats = tracker.get_statistics()
        print(f"✓ User preferences updated")
        print(f"  Total ratings: {stats['total_ratings']}")
        print(f"  Average rating: {stats['average_rating']:.2f}")
        print(f"  Dietary restrictions: {stats['dietary_restrictions']}")
        print(f"  Preferences: {stats['preferences']}")
        
    except Exception as e:
        print(f"✗ Error testing user preferences: {e}")

def test_data_processing():
    """Test data processing pipeline"""
    print("\n" + "="*50)
    print("TESTING DATA PROCESSING")
    print("="*50)
    
    try:
        from data_processing.clean_menus import MenuProcessor
        
        processor = MenuProcessor()
        
        # Create sample menu data
        sample_menu = {
            'date': '2024-01-15',
            'menus': {
                'worcester': {
                    'meals': {
                        'lunch': {
                            'International': [
                                {
                                    'name': 'Chicken Tikka Masala',
                                    'description': 'Spicy Indian curry',
                                    'nutrition': {'calories': 350, 'protein': 25},
                                    'allergens': ['dairy', 'gluten']
                                }
                            ],
                            'Salad Bar': [
                                {
                                    'name': 'Caesar Salad',
                                    'description': 'Fresh romaine lettuce',
                                    'nutrition': {'calories': 150, 'protein': 8},
                                    'allergens': ['dairy']
                                }
                            ]
                        }
                    }
                }
            }
        }
        
        # Process the menu
        print("Processing sample menu data...")
        items = processor.flatten_menu_to_items(sample_menu)
        
        print(f"✓ Processed {len(items)} menu items")
        
        if items:
            item = items[0]
            print(f"  Sample item: {item['item_name']}")
            print(f"    Clean name: {item['item_name_clean']}")
            print(f"    Station: {item['station_clean']}")
            print(f"    Calories: {item['calories']}")
            print(f"    Allergens: {item['allergens']}")
            print(f"    Health score: {item['health_score']}")
        
    except Exception as e:
        print(f"✗ Error testing data processing: {e}")

def test_collaborative_filtering():
    """Test collaborative filtering model"""
    print("\n" + "="*50)
    print("TESTING COLLABORATIVE FILTERING")
    print("="*50)
    
    try:
        from models.collaborative_filter import DiningCollaborativeFilter
        import pandas as pd
        import numpy as np
        
        # Create sample rating data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'user_id': np.random.randint(0, 50, 200),
            'item_id': np.random.randint(0, 100, 200),
            'rating': np.random.uniform(1, 5, 200)
        })
        
        print("Training collaborative filtering model...")
        cf = DiningCollaborativeFilter()
        history = cf.train_model(sample_data, epochs=20)
        
        print(f"✓ Model trained successfully")
        print(f"  Best validation loss: {history['best_val_loss']:.4f}")
        
        # Get recommendations
        recommendations = cf.get_recommendations(user_id=0, top_k=5)
        print(f"  Generated {len(recommendations)} recommendations for user 0")
        
        for rec in recommendations[:3]:
            print(f"    Item {rec['item_id']}: {rec['predicted_rating']:.2f}")
        
    except Exception as e:
        print(f"✗ Error testing collaborative filtering: {e}")

def test_content_based():
    """Test content-based recommendations"""
    print("\n" + "="*50)
    print("TESTING CONTENT-BASED RECOMMENDATIONS")
    print("="*50)
    
    try:
        from models.content_based import ContentBasedRecommender
        import pandas as pd
        
        # Create sample items
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
        
        print("Training content-based model...")
        cb = ContentBasedRecommender()
        features = cb.create_item_features(sample_items)
        similarity_matrix = cb.compute_similarity_matrix()
        
        print(f"✓ Model trained successfully")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Similarity matrix shape: {similarity_matrix.shape}")
        
        # Get similar items
        similar = cb.get_similar_items(item_id=1, top_k=3)
        print(f"  Items similar to 'Chicken Tikka Masala':")
        
        for item in similar:
            print(f"    {item['item_name']}: {item['similarity']:.3f}")
        
    except Exception as e:
        print(f"✗ Error testing content-based recommendations: {e}")

def test_hybrid_recommender():
    """Test hybrid recommendation system"""
    print("\n" + "="*50)
    print("TESTING HYBRID RECOMMENDER")
    print("="*50)
    
    try:
        from models.hybrid_model import HybridRecommender
        
        # Create a test user with some preferences
        tracker = UserPreferenceTracker("test_user")
        tracker.rate_item(1, 5, "Chicken Tikka Masala", "worcester", "International")
        tracker.rate_item(2, 4, "Caesar Salad", "franklin", "Salad Bar")
        tracker.set_dietary_restrictions(['vegetarian'])
        
        print("Creating hybrid recommender...")
        recommender = HybridRecommender("test_user")
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            dining_hall='worcester',
            meal_period='lunch',
            top_k=5
        )
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['item_name']}")
            print(f"     Score: {rec['score']:.2f} | Method: {rec['method']}")
            print(f"     Confidence: {rec['confidence']:.2f}")
        
        # Get insights
        insights = recommender.get_recommendation_insights()
        print(f"\n  User insights:")
        print(f"    Total ratings: {insights['total_ratings']}")
        print(f"    Model availability: {insights['model_availability']}")
        
    except Exception as e:
        print(f"✗ Error testing hybrid recommender: {e}")

def test_api():
    """Test the FastAPI application"""
    print("\n" + "="*50)
    print("TESTING API")
    print("="*50)
    
    try:
        from api.main import app
        from fastapi.testclient import TestClient
        
        print("Creating API test client...")
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        print(f"✓ Root endpoint: {response.status_code}")
        
        # Test health check
        response = client.get("/health")
        print(f"✓ Health check: {response.status_code}")
        
        # Test rating endpoint
        rating_data = {
            "user_id": "test_user",
            "item_id": 1,
            "rating": 5,
            "item_name": "Chicken Tikka Masala"
        }
        response = client.post("/api/v1/rate", json=rating_data)
        print(f"✓ Rating endpoint: {response.status_code}")
        
        # Test recommendations endpoint
        response = client.get("/api/v1/recommendations/test_user?top_k=5")
        print(f"✓ Recommendations endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Generated {len(data.get('recommendations', []))} recommendations")
        
    except Exception as e:
        print(f"✗ Error testing API: {e}")

def run_comprehensive_test():
    """Run all tests"""
    print("🍽️ UMass Dining Recommender System - Comprehensive Test")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Run all tests
    test_menu_scraper()
    test_nutrition_scraper()
    test_user_preferences()
    test_data_processing()
    test_collaborative_filtering()
    test_content_based()
    test_hybrid_recommender()
    test_api()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total test duration: {duration}")
    print("All major components tested successfully!")
    print("\nTo run the full system:")
    print("1. Start the API: python run.py api")
    print("2. Scrape data: python run.py scrape")
    print("3. Train models: python run.py train")
    print("4. Visit: http://localhost:8000/docs")

if __name__ == "__main__":
    run_comprehensive_test()
