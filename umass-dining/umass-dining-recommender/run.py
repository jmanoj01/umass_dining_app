#!/usr/bin/env python3
"""
UMass Dining Recommender - Main Entry Point

This script provides a command-line interface for running different components
of the dining recommendation system.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from scrapers.menu_scraper import UMassDiningScraper
from scrapers.nutrition_scraper import NutritionScraper
from scrapers.scheduler import DiningDataScheduler
from models.user_preferences import UserPreferenceTracker
from models.collaborative_filter import DiningCollaborativeFilter
from models.content_based import ContentBasedRecommender
from api.main import app
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_scraper(args):
    """Run the menu scraper"""
    logger.info("Starting menu scraper...")
    
    scraper = UMassDiningScraper()
    
    if args.historical:
        logger.info(f"Scraping historical data for {args.days} days...")
        scraper.scrape_historical_menus(args.days)
    else:
        logger.info("Scraping today's menus...")
        menus = scraper.scrape_all_dining_halls()
        if menus:
            scraper.save_menu_data(menus)
            logger.info("Menu scraping completed successfully")
        else:
            logger.warning("No menu data was scraped")

def run_nutrition_scraper(args):
    """Run the nutrition scraper"""
    logger.info("Starting nutrition scraper...")
    
    scraper = NutritionScraper()
    
    # Load sample items for testing
    sample_items = [
        {'name': 'Chicken Breast', 'description': 'Grilled chicken breast'},
        {'name': 'Caesar Salad', 'description': 'Fresh romaine lettuce with caesar dressing'},
        {'name': 'Pizza Slice', 'description': 'Cheese pizza slice'},
        {'name': 'French Fries', 'description': 'Crispy golden french fries'}
    ]
    
    results = scraper.batch_scrape_nutrition(sample_items)
    filename = scraper.save_nutrition_data(results)
    logger.info(f"Nutrition data saved to {filename}")

def run_scheduler(args):
    """Run the automated scheduler"""
    logger.info("Starting automated scheduler...")
    
    scheduler = DiningDataScheduler()
    
    if args.manual:
        scheduler.run_manual_scrape()
    else:
        scheduler.start()
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()

def run_api(args):
    """Run the FastAPI server"""
    logger.info("Starting API server...")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

def run_user_demo(args):
    """Run user preference demo"""
    logger.info("Running user preference demo...")
    
    tracker = UserPreferenceTracker(args.user_id)
    
    # Demo: Rate some items
    tracker.rate_item(
        item_id=1, 
        rating=5, 
        item_name="Chicken Tikka Masala",
        dining_hall="worcester",
        station="International"
    )
    
    tracker.rate_item(
        item_id=2, 
        rating=3, 
        item_name="Caesar Salad",
        dining_hall="franklin",
        station="Salad Bar"
    )
    
    # Set preferences
    tracker.set_dietary_restrictions(['vegetarian'])
    tracker.set_preferences({
        'spice_level': 'high',
        'portion_size': 'large',
        'cuisine_preferences': ['indian', 'mexican']
    })
    
    # Add to history
    tracker.add_to_history(
        item_id=1,
        dining_hall='worcester',
        meal_period='dinner',
        item_name="Chicken Tikka Masala",
        station="International"
    )
    
    # Show stats
    stats = tracker.get_statistics()
    print("\nUser Statistics:")
    print(f"Total ratings: {stats['total_ratings']}")
    print(f"Average rating: {stats['average_rating']:.2f}")
    print(f"Dietary restrictions: {stats['dietary_restrictions']}")
    print(f"Preferences: {stats['preferences']}")

def run_model_training(args):
    """Run model training"""
    logger.info("Starting model training...")
    
    if args.model == "collaborative" or args.model == "all":
        logger.info("Training collaborative filtering model...")
        cf = DiningCollaborativeFilter()
        
        # Create sample data for demo
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'user_id': np.random.randint(0, 100, 1000),
            'item_id': np.random.randint(0, 200, 1000),
            'rating': np.random.uniform(1, 5, 1000)
        })
        
        history = cf.train_model(sample_data, epochs=50)
        model_file = cf.save_model()
        logger.info(f"Collaborative model saved to {model_file}")
    
    if args.model == "content" or args.model == "all":
        logger.info("Training content-based model...")
        cb = ContentBasedRecommender()
        
        # Create sample items for demo
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
        
        features = cb.create_item_features(sample_items)
        similarity_matrix = cb.compute_similarity_matrix()
        model_file = cb.save_model()
        logger.info(f"Content-based model saved to {model_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="UMass Dining Recommender - AI-powered dining recommendations"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scraper command
    scraper_parser = subparsers.add_parser('scrape', help='Scrape menu data')
    scraper_parser.add_argument('--historical', action='store_true', help='Scrape historical data')
    scraper_parser.add_argument('--days', type=int, default=7, help='Number of days to scrape')
    
    # Nutrition scraper command
    nutrition_parser = subparsers.add_parser('nutrition', help='Scrape nutrition data')
    
    # Scheduler command
    scheduler_parser = subparsers.add_parser('schedule', help='Run automated scheduler')
    scheduler_parser.add_argument('--manual', action='store_true', help='Run manual scrape')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Run API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    api_parser.add_argument('--log-level', default='info', help='Log level')
    
    # User demo command
    user_parser = subparsers.add_parser('user-demo', help='Run user preference demo')
    user_parser.add_argument('--user-id', default='demo_user', help='User ID for demo')
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Train recommendation models')
    train_parser.add_argument('--model', choices=['collaborative', 'content', 'all'], 
                             default='all', help='Model to train')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'scrape':
            run_scraper(args)
        elif args.command == 'nutrition':
            run_nutrition_scraper(args)
        elif args.command == 'schedule':
            run_scheduler(args)
        elif args.command == 'api':
            run_api(args)
        elif args.command == 'user-demo':
            run_user_demo(args)
        elif args.command == 'train':
            run_model_training(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
