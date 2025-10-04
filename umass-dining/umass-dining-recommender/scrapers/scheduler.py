import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scrapers.menu_scraper import UMassDiningScraper
from scrapers.nutrition_scraper import NutritionScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiningDataScheduler:
    """
    Automated scheduler for scraping UMass dining data
    Runs daily scraping tasks and maintains data freshness
    """
    
    def __init__(self, config_file: str = "config/scheduler_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize scrapers
        self.menu_scraper = UMassDiningScraper()
        self.nutrition_scraper = NutritionScraper()
        
        # Create necessary directories
        self.setup_directories()
        
        # Scheduler state
        self.is_running = False
        self.last_run = None
        self.run_count = 0
        self.error_count = 0
        
        # Setup scheduled jobs
        self.setup_schedule()
    
    def load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration"""
        default_config = {
            "scraping": {
                "daily_menu_time": "06:00",  # 6 AM daily
                "nutrition_update_time": "02:00",  # 2 AM daily
                "historical_days": 7,  # Scrape last 7 days
                "retry_failed": True,
                "max_retries": 3
            },
            "data_retention": {
                "keep_days": 30,  # Keep 30 days of data
                "cleanup_time": "03:00"  # 3 AM daily cleanup
            },
            "notifications": {
                "email_alerts": False,
                "log_level": "INFO"
            },
            "dining_halls": {
                "enabled": ["worcester", "franklin", "berkshire", "hampshire"],
                "priority": ["worcester", "franklin"]  # Try these first
            }
        }
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        # Save default config
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            "logs",
            "data/raw/menus",
            "data/processed",
            "data/embeddings",
            "config",
            "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_schedule(self):
        """Setup scheduled jobs"""
        # Daily menu scraping
        schedule.every().day.at(self.config['scraping']['daily_menu_time']).do(
            self.daily_menu_scrape
        )
        
        # Nutrition data updates
        schedule.every().day.at(self.config['scraping']['nutrition_update_time']).do(
            self.update_nutrition_data
        )
        
        # Data cleanup
        schedule.every().day.at(self.config['data_retention']['cleanup_time']).do(
            self.cleanup_old_data
        )
        
        # Health check every hour
        schedule.every().hour.do(self.health_check)
        
        logger.info("Scheduled jobs configured:")
        logger.info(f"  - Daily menu scraping: {self.config['scraping']['daily_menu_time']}")
        logger.info(f"  - Nutrition updates: {self.config['scraping']['nutrition_update_time']}")
        logger.info(f"  - Data cleanup: {self.config['data_retention']['cleanup_time']}")
    
    def daily_menu_scrape(self):
        """Daily menu scraping job"""
        logger.info("Starting daily menu scrape...")
        start_time = datetime.now()
        
        try:
            # Scrape today's menus
            menus = self.menu_scraper.scrape_all_dining_halls()
            
            if menus:
                filename = self.menu_scraper.save_menu_data(menus)
                logger.info(f"Daily scrape completed: {filename}")
                
                # Update nutrition data for new items
                self.update_nutrition_for_new_items(menus)
                
                self.run_count += 1
                self.last_run = datetime.now()
                
                # Backup successful run
                self.create_backup()
                
            else:
                logger.warning("No menu data scraped today")
                self.error_count += 1
                
        except Exception as e:
            logger.error(f"Daily menu scrape failed: {e}")
            self.error_count += 1
            
            # Retry if configured
            if self.config['scraping']['retry_failed']:
                self.retry_failed_scrape()
        
        duration = datetime.now() - start_time
        logger.info(f"Daily scrape completed in {duration}")
    
    def update_nutrition_data(self):
        """Update nutrition data for existing items"""
        logger.info("Starting nutrition data update...")
        
        try:
            # Load recent menu data
            recent_menus = self.get_recent_menu_data(days=7)
            
            if not recent_menus:
                logger.warning("No recent menu data found for nutrition update")
                return
            
            # Extract unique items
            unique_items = self.extract_unique_items(recent_menus)
            
            # Update nutrition data
            nutrition_results = self.nutrition_scraper.batch_scrape_nutrition(unique_items)
            
            # Save updated nutrition data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/processed/nutrition_update_{timestamp}.json"
            self.nutrition_scraper.save_nutrition_data(nutrition_results, filename)
            
            logger.info(f"Nutrition update completed: {filename}")
            
        except Exception as e:
            logger.error(f"Nutrition update failed: {e}")
    
    def update_nutrition_for_new_items(self, menus: Dict):
        """Update nutrition data for newly scraped items"""
        try:
            # Extract items from today's menus
            items = []
            for hall, menu in menus.items():
                for meal, stations in menu['meals'].items():
                    for station, station_items in stations.items():
                        for item in station_items:
                            items.append({
                                'name': item['name'],
                                'description': item.get('description', '')
                            })
            
            if items:
                # Scrape nutrition for new items
                nutrition_results = self.nutrition_scraper.batch_scrape_nutrition(items)
                
                # Save to daily nutrition file
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"data/processed/daily_nutrition_{timestamp}.json"
                self.nutrition_scraper.save_nutrition_data(nutrition_results, filename)
                
                logger.info(f"Updated nutrition for {len(items)} items")
                
        except Exception as e:
            logger.error(f"Failed to update nutrition for new items: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data files"""
        logger.info("Starting data cleanup...")
        
        try:
            keep_days = self.config['data_retention']['keep_days']
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            # Clean up old menu files
            menu_dir = Path("data/raw/menus")
            deleted_count = 0
            
            for file_path in menu_dir.glob("menu_*.json"):
                try:
                    # Extract date from filename
                    date_str = file_path.stem.split('_')[1]  # menu_YYYYMMDD_timestamp
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
            
            # Clean up old nutrition files
            nutrition_dir = Path("data/processed")
            for file_path in nutrition_dir.glob("nutrition_*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
            
            logger.info(f"Cleanup completed: deleted {deleted_count} old files")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def health_check(self):
        """Perform health check and log status"""
        try:
            # Check disk space
            disk_usage = self.get_disk_usage()
            
            # Check recent data availability
            recent_data = self.get_recent_menu_data(days=1)
            
            # Log health status
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'last_run': self.last_run.isoformat() if self.last_run else None,
                'run_count': self.run_count,
                'error_count': self.error_count,
                'disk_usage_gb': disk_usage,
                'recent_data_available': len(recent_data) > 0,
                'recent_data_count': len(recent_data)
            }
            
            logger.info(f"Health check: {json.dumps(status, indent=2)}")
            
            # Alert if issues detected
            if self.error_count > 5:
                logger.warning("High error count detected!")
            
            if disk_usage > 5.0:  # More than 5GB
                logger.warning("High disk usage detected!")
            
            if not recent_data:
                logger.warning("No recent data available!")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_disk_usage(self) -> float:
        """Get disk usage in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            return used / (1024**3)  # Convert to GB
        except:
            return 0.0
    
    def get_recent_menu_data(self, days: int = 7) -> List[Dict]:
        """Get recent menu data files"""
        recent_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        menu_dir = Path("data/raw/menus")
        for file_path in menu_dir.glob("menu_*.json"):
            try:
                if file_path.stat().st_mtime >= cutoff_date.timestamp():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        recent_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        return recent_data
    
    def extract_unique_items(self, menu_data: List[Dict]) -> List[Dict]:
        """Extract unique items from menu data"""
        unique_items = {}
        
        for menu in menu_data:
            for hall, hall_data in menu.get('menus', {}).items():
                for meal, stations in hall_data.get('meals', {}).items():
                    for station, items in stations.items():
                        for item in items:
                            name = item['name'].lower().strip()
                            if name not in unique_items:
                                unique_items[name] = {
                                    'name': item['name'],
                                    'description': item.get('description', '')
                                }
        
        return list(unique_items.values())
    
    def retry_failed_scrape(self):
        """Retry failed scraping operations"""
        logger.info("Retrying failed scrape...")
        
        max_retries = self.config['scraping']['max_retries']
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                
                # Try scraping with different dining halls
                priority_halls = self.config['dining_halls']['priority']
                menus = {}
                
                for hall in priority_halls:
                    menu = self.menu_scraper.scrape_daily_menu(hall)
                    if menu:
                        menus[hall] = menu
                        time.sleep(2)  # Be respectful
                
                if menus:
                    self.menu_scraper.save_menu_data(menus)
                    logger.info("Retry successful!")
                    return
                else:
                    logger.warning(f"Retry attempt {attempt + 1} failed")
                    time.sleep(60)  # Wait before next retry
                    
            except Exception as e:
                logger.error(f"Retry attempt {attempt + 1} failed: {e}")
                time.sleep(60)
        
        logger.error("All retry attempts failed")
    
    def create_backup(self):
        """Create backup of current data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(f"backups/backup_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy recent data files
            import shutil
            
            # Copy menu data
            menu_dir = Path("data/raw/menus")
            if menu_dir.exists():
                shutil.copytree(menu_dir, backup_dir / "menus")
            
            # Copy processed data
            processed_dir = Path("data/processed")
            if processed_dir.exists():
                shutil.copytree(processed_dir, backup_dir / "processed")
            
            logger.info(f"Backup created: {backup_dir}")
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        logger.info("Starting dining data scheduler...")
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            self.is_running = False
    
    def start(self):
        """Start the scheduler in background thread"""
        if not self.is_running:
            scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            scheduler_thread.start()
            logger.info("Scheduler started in background")
        else:
            logger.warning("Scheduler is already running")
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        logger.info("Scheduler stopped")
    
    def run_manual_scrape(self):
        """Run a manual scrape immediately"""
        logger.info("Running manual scrape...")
        self.daily_menu_scrape()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'run_count': self.run_count,
            'error_count': self.error_count,
            'config': self.config
        }

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='UMass Dining Data Scheduler')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--manual', action='store_true', help='Run manual scrape')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    scheduler = DiningDataScheduler(args.config)
    
    if args.start:
        scheduler.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
    elif args.manual:
        scheduler.run_manual_scrape()
    elif args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
    else:
        print("Use --help for usage information")
