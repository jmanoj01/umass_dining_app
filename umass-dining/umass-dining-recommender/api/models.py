from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class RatingRequest(BaseModel):
    """Request model for rating an item"""
    user_id: str
    item_id: int
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1 to 5")
    item_name: Optional[str] = None
    dining_hall: Optional[str] = None
    station: Optional[str] = None

class HistoryRequest(BaseModel):
    """Request model for adding to eating history"""
    user_id: str
    item_id: int
    dining_hall: str
    meal_period: str
    item_name: Optional[str] = None
    station: Optional[str] = None
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)

class SearchRequest(BaseModel):
    """Request model for searching items"""
    criteria: Dict[str, Any]
    top_k: int = Field(10, ge=1, le=100)

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: str
    algorithm: str
    recommendations: List[Dict[str, Any]]
    count: int
    timestamp: str

class ItemProfile(BaseModel):
    """Model for item profile"""
    item_id: int
    features: Dict[str, Any]
    nutritional_info: Dict[str, float]
    dietary_info: Dict[str, bool]
    health_score: int

class UserStats(BaseModel):
    """Model for user statistics"""
    user_id: str
    total_ratings: int
    average_rating: float
    total_meals_tracked: int
    dietary_restrictions: List[str]
    preferences: Dict[str, Any]
    analytics: Dict[str, Any]

class SimilarItem(BaseModel):
    """Model for similar items"""
    item_id: int
    similarity: float
    confidence: float

class SearchResult(BaseModel):
    """Model for search results"""
    item_id: int
    health_score: int
    calories: float
    protein: float

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    message: str
    timestamp: str

class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]

class MenuItem(BaseModel):
    """Model for menu items"""
    name: str
    description: Optional[str] = None
    nutrition: Dict[str, Any] = {}
    allergens: List[str] = []
    station: Optional[str] = None
    dining_hall: Optional[str] = None
    meal_period: Optional[str] = None

class DiningHall(BaseModel):
    """Model for dining halls"""
    id: str
    name: str
    location: Optional[str] = None
    hours: Optional[Dict[str, str]] = None

class MealPeriod(BaseModel):
    """Model for meal periods"""
    name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    items: List[MenuItem] = []

class MenuResponse(BaseModel):
    """Response model for menu data"""
    dining_hall: str
    date: str
    meals: Dict[str, Dict[str, List[MenuItem]]]
    scraped_at: str

class UserPreferences(BaseModel):
    """Model for user preferences"""
    user_id: str
    dietary_restrictions: List[str] = []
    favorite_stations: List[str] = []
    meal_times: List[str] = []
    cuisine_preferences: List[str] = []
    spice_level: str = "medium"
    portion_size: str = "medium"

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    user_id: str
    algorithm: str = "hybrid"
    top_k: int = Field(10, ge=1, le=50)
    include_rated: bool = False
    filters: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    """Request model for training models"""
    model_type: str = Field(..., pattern="^(collaborative|content|hybrid)$")
    data_file: Optional[str] = None
    epochs: int = Field(100, ge=1, le=1000)
    parameters: Optional[Dict[str, Any]] = None

class ModelStatus(BaseModel):
    """Model for model status"""
    model_type: str
    is_trained: bool
    last_trained: Optional[str] = None
    accuracy: Optional[float] = None
    parameters: Dict[str, Any] = {}

class AnalyticsResponse(BaseModel):
    """Response model for analytics"""
    total_users: int
    total_items: int
    total_ratings: int
    average_rating: float
    most_popular_items: List[Dict[str, Any]]
    most_active_users: List[Dict[str, Any]]
    model_performance: Dict[str, float]

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    user_id: str
    recommendation_id: Optional[str] = None
    feedback_type: str = Field(..., pattern="^(helpful|not_helpful|irrelevant)$")
    comment: Optional[str] = None
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)

class NotificationRequest(BaseModel):
    """Request model for notifications"""
    user_id: str
    notification_type: str = Field(..., pattern="^(recommendation|reminder|update)$")
    message: str
    data: Optional[Dict[str, Any]] = None

class ExportRequest(BaseModel):
    """Request model for data export"""
    user_id: str
    format: str = Field("json", pattern="^(json|csv)$")
    include_history: bool = True
    include_ratings: bool = True

class ImportRequest(BaseModel):
    """Request model for data import"""
    user_id: str
    data: Dict[str, Any]
    overwrite: bool = False

class BatchRatingRequest(BaseModel):
    """Request model for batch rating"""
    user_id: str
    ratings: List[RatingRequest]

class BatchHistoryRequest(BaseModel):
    """Request model for batch history"""
    user_id: str
    history: List[HistoryRequest]

class ModelMetrics(BaseModel):
    """Model for model metrics"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    timestamp: str

class SystemStatus(BaseModel):
    """Model for system status"""
    status: str
    uptime: str
    memory_usage: float
    cpu_usage: float
    active_models: List[str]
    last_update: str

class ConfigurationRequest(BaseModel):
    """Request model for configuration updates"""
    config_type: str
    parameters: Dict[str, Any]
    apply_immediately: bool = True

class LogEntry(BaseModel):
    """Model for log entries"""
    timestamp: str
    level: str
    message: str
    user_id: Optional[str] = None
    action: Optional[str] = None

class AuditTrail(BaseModel):
    """Model for audit trail"""
    user_id: str
    action: str
    resource: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class PerformanceMetrics(BaseModel):
    """Model for performance metrics"""
    endpoint: str
    response_time: float
    status_code: int
    timestamp: str
    user_id: Optional[str] = None

class CacheStats(BaseModel):
    """Model for cache statistics"""
    cache_type: str
    hit_rate: float
    miss_rate: float
    total_requests: int
    cache_size: int

class DatabaseStats(BaseModel):
    """Model for database statistics"""
    total_records: int
    table_sizes: Dict[str, int]
    last_backup: Optional[str] = None
    connection_pool: Dict[str, int]

class SecurityEvent(BaseModel):
    """Model for security events"""
    event_type: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: str
    severity: str
    description: str

class MaintenanceWindow(BaseModel):
    """Model for maintenance windows"""
    start_time: str
    end_time: str
    description: str
    affected_services: List[str]
    status: str = "scheduled"
