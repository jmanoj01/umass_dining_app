import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    api_reload: bool = True
    
    # Database Configuration
    database_url: str = "sqlite:///./data/dining_recommender.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Model Configuration
    model_update_interval: int = 3600  # seconds
    collaborative_factors: int = 50
    content_similarity_threshold: float = 0.3
    hybrid_collaborative_weight: float = 0.6
    hybrid_content_weight: float = 0.4
    
    # Scraping Configuration
    scraping_enabled: bool = True
    scraping_interval: int = 3600  # seconds
    scraping_retry_attempts: int = 3
    scraping_delay: int = 2  # seconds between requests
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/dining_recommender.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Security Configuration
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    
    # Cache Configuration
    cache_ttl: int = 3600  # seconds
    cache_max_size: int = 1000
    
    # Notification Configuration
    notifications_enabled: bool = True
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = "your-email@gmail.com"
    email_password: str = "your-app-password"
    
    # Analytics Configuration
    analytics_enabled: bool = True
    analytics_retention_days: int = 90
    
    # Performance Configuration
    max_recommendations: int = 50
    recommendation_cache_ttl: int = 1800  # seconds
    batch_size: int = 32
    
    # Data Retention
    data_retention_days: int = 365
    backup_interval: int = 86400  # seconds (24 hours)
    
    # Development Configuration
    debug: bool = True
    testing: bool = False
    mock_data: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create necessary directories for the application"""
    directories = [
        "data/raw/menus",
        "data/processed",
        "data/embeddings",
        "models",
        "logs",
        "backups",
        "user_data",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()
