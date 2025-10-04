#!/usr/bin/env python3
"""
UMass Dining Recommender System - Complete Demo Script

This script demonstrates the full functionality of the UMass Dining Recommender System,
including data collection, processing, model training, and API usage.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def wait_for_api(api_url="http://localhost:8000", max_attempts=30):
    """Wait for the API to be available"""
    print("Waiting for API to be available...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {attempt + 1}/{max_attempts} - API not ready yet...")
        time.sleep(2)
    
    print("❌ API failed to start within expected time")
    return False

def demo_data_collection():
    """Demonstrate data collection"""
    print_step("1", "Data Collection Demo")
    
    try:
        from scrapers.menu_scraper import UMassDiningScraper
        
        print("Creating sample menu data...")
        scraper = UMassDiningScraper()
        
        # Create sample data instead of actually scraping
        sample_data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "scraped_at": datetime.now().isoformat(),
            "menus": {
                "worcester": {
                    "dining_hall": "worcester",
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "meals": {
                        "breakfast": {
                            "Main Station": [
                                {
                                    "name": "Scrambled Eggs",
                                    "description": "Fresh scrambled eggs",
                                    "nutrition": {"calories": "200", "protein": "12g"},
                                    "allergens": ["eggs"]
                                },
                                {
                                    "name": "Pancakes",
                                    "description": "Fluffy pancakes with syrup",
                                    "nutrition": {"calories": "300", "protein": "8g"},
                                    "allergens": ["gluten", "dairy"]
                                }
                            ],
                            "Cereal Station": [
                                {
                                    "name": "Oatmeal",
                                    "description": "Steel-cut oats",
                                    "nutrition": {"calories": "150", "protein": "5g"},
                                    "allergens": []
                                }
                            ]
                        },
                        "lunch": {
                            "Main Station": [
                                {
                                    "name": "Grilled Chicken Breast",
                                    "description": "Herb-seasoned chicken",
                                    "nutrition": {"calories": "250", "protein": "30g"},
                                    "allergens": []
                                },
                                {
                                    "name": "Caesar Salad",
                                    "description": "Fresh romaine with caesar dressing",
                                    "nutrition": {"calories": "180", "protein": "6g"},
                                    "allergens": ["dairy", "eggs"]
                                }
                            ],
                            "Pizza Station": [
                                {
                                    "name": "Cheese Pizza",
                                    "description": "Classic cheese pizza",
                                    "nutrition": {"calories": "400", "protein": "15g"},
                                    "allergens": ["gluten", "dairy"]
                                }
                            ]
                        },
                        "dinner": {
                            "Main Station": [
                                {
                                    "name": "Salmon Fillet",
                                    "description": "Baked salmon with herbs",
                                    "nutrition": {"calories": "300", "protein": "35g"},
                                    "allergens": ["fish"]
                                },
                                {
                                    "name": "Vegetable Stir Fry",
                                    "description": "Mixed vegetables in soy sauce",
                                    "nutrition": {"calories": "120", "protein": "4g"},
                                    "allergens": ["soy"]
                                }
                            ]
                        }
                    }
                },
                "franklin": {
                    "dining_hall": "franklin",
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "meals": {
                        "lunch": {
                            "Main Station": [
                                {
                                    "name": "Turkey Sandwich",
                                    "description": "Fresh turkey on whole wheat",
                                    "nutrition": {"calories": "350", "protein": "20g"},
                                    "allergens": ["gluten"]
                                }
                            ]
                        }
                    }
                }
            }
        }
        
        # Save sample data
        os.makedirs("data/raw/menus", exist_ok=True)
        filename = f"data/raw/menus/menu_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✓ Sample menu data saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False

def demo_data_processing():
    """Demonstrate data processing"""
    print_step("2", "Data Processing Demo")
    
    try:
        from data_processing.clean_menus import MenuProcessor
        
        print("Processing menu data...")
        processor = MenuProcessor()
        
        # Process the sample data
        all_items_df, unique_items_df = processor.create_item_database()
        
        print(f"✓ Processed {len(all_items_df)} menu items")
        print(f"✓ Created database of {len(unique_items_df)} unique items")
        
        # Show some sample items
        print("\nSample unique items:")
        for _, item in unique_items_df.head(5).iterrows():
            print(f"  - {item['item_name']} (ID: {item['item_id']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False

def demo_embeddings():
    """Demonstrate embedding generation"""
    print_step("3", "Embedding Generation Demo")
    
    try:
        from data_processing.item_embeddings import FoodEmbeddings
        
        print("Generating food embeddings...")
        embedder = FoodEmbeddings()
        
        # Generate embeddings
        embeddings, df = embedder.generate_embeddings()
        
        print(f"✓ Generated embeddings for {len(embeddings)} items")
        
        # Test similarity search
        print("\nTesting similarity search...")
        similar = embedder.find_similar_items("chicken breast", top_k=3)
        if similar:
            print("Items similar to 'chicken breast':")
            for item in similar:
                print(f"  - {item['item_name']}: {item['similarity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in embedding generation: {e}")
        return False

def demo_user_preferences():
    """Demonstrate user preference tracking"""
    print_step("4", "User Preference Tracking Demo")
    
    try:
        from models.user_preferences import UserPreferenceTracker
        
        print("Creating user preference tracker...")
        tracker = UserPreferenceTracker("demo_user")
        
        # Add some sample ratings
        sample_ratings = [
            (1, 5, "Scrambled Eggs"),
            (2, 4, "Pancakes"),
            (3, 3, "Oatmeal"),
            (4, 5, "Grilled Chicken Breast"),
            (5, 2, "Caesar Salad"),
            (6, 4, "Cheese Pizza"),
            (7, 5, "Salmon Fillet"),
            (8, 3, "Vegetable Stir Fry")
        ]
        
        print("Adding sample ratings...")
        for item_id, rating, item_name in sample_ratings:
            tracker.rate_item(item_id, rating, item_name)
        
        # Set dietary preferences
        tracker.set_dietary_restrictions(['vegetarian'])
        
        # Add to history
        tracker.add_to_history(1, 'worcester', 'breakfast', 'Scrambled Eggs')
        tracker.add_to_history(4, 'worcester', 'lunch', 'Grilled Chicken Breast')
        
        # Show statistics
        stats = tracker.get_statistics()
        print(f"✓ User has {stats['total_ratings']} ratings")
        print(f"✓ Average rating: {stats['average_rating']:.1f}")
        print(f"✓ Meals tracked: {stats['total_meals_tracked']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in user preferences: {e}")
        return False

def demo_recommendations():
    """Demonstrate recommendation system"""
    print_step("5", "Recommendation System Demo")
    
    try:
        from models.hybrid_model import HybridRecommender
        
        print("Creating hybrid recommender...")
        recommender = HybridRecommender("demo_user")
        
        # Get recommendations
        print("Getting recommendations...")
        recommendations = recommender.get_recommendations(
            dining_hall='worcester',
            meal_period='lunch',
            top_k=5
        )
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['item_name']}")
            print(f"     Score: {rec['score']:.3f} | Method: {rec['method']}")
            print(f"     Station: {rec['station']} | Calories: {rec.get('calories', 'N/A')}")
            
            # Get explanation
            explanations = recommender.explain_recommendation(rec['item_id'])
            if explanations:
                print(f"     Why: {' | '.join(explanations)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in recommendations: {e}")
        return False

def demo_api():
    """Demonstrate API usage"""
    print_step("6", "API Demo")
    
    api_url = "http://localhost:8000"
    
    # Wait for API to be ready
    if not wait_for_api(api_url):
        print("Skipping API demo - API not available")
        return False
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print("✓ API is healthy")
        else:
            print("❌ API health check failed")
            return False
        
        # Test recommendations endpoint
        print("Testing recommendations endpoint...")
        response = requests.get(f"{api_url}/api/v1/recommendations/demo_user")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Got {len(data.get('recommendations', []))} recommendations")
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
        
        # Test rating endpoint
        print("Testing rating endpoint...")
        rating_data = {
            "user_id": "demo_user",
            "item_id": 1,
            "rating": 5,
            "item_name": "Scrambled Eggs"
        }
        response = requests.post(f"{api_url}/api/v1/rate", json=rating_data)
        if response.status_code == 200:
            print("✓ Successfully rated item")
        else:
            print(f"❌ Rating failed: {response.status_code}")
        
        # Test user stats endpoint
        print("Testing user stats endpoint...")
        response = requests.get(f"{api_url}/api/v1/user/demo_user/stats")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print(f"✓ User stats: {stats.get('total_ratings', 0)} ratings, avg {stats.get('average_rating', 0):.1f}")
        else:
            print(f"❌ User stats failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API demo: {e}")
        return False

def main():
    """Run the complete demo"""
    print_header("UMass Dining Recommender System - Complete Demo")
    
    print("This demo will showcase all components of the UMass Dining Recommender System:")
    print("1. Data Collection (Web Scraping)")
    print("2. Data Processing (Cleaning & Standardization)")
    print("3. Embedding Generation (NLP)")
    print("4. User Preference Tracking")
    print("5. Recommendation System (Hybrid Model)")
    print("6. API Integration")
    
    input("\nPress Enter to start the demo...")
    
    # Run all demos
    demos = [
        demo_data_collection,
        demo_data_processing,
        demo_embeddings,
        demo_user_preferences,
        demo_recommendations,
        demo_api
    ]
    
    success_count = 0
    for demo in demos:
        try:
            if demo():
                success_count += 1
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
    
    # Summary
    print_header("Demo Summary")
    print(f"Completed {success_count}/{len(demos)} demos successfully")
    
    if success_count == len(demos):
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python run.py api")
        print("2. Start the frontend: cd ../umass-dining-frontend && npm run dev")
        print("3. Visit http://localhost:3000 to see the web interface")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    print("\nThank you for trying the UMass Dining Recommender System!")

if __name__ == "__main__":
    main()
