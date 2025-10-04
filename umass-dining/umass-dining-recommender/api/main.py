from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.routes import router
from api.models import *
from models.user_preferences import UserPreferenceTracker
from models.collaborative_filter import DiningCollaborativeFilter
from models.content_based import ContentBasedRecommender

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="UMass Dining Recommender API",
    description="AI-powered dining recommendations for UMass students",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom error handlers
from api.errors import AppError, app_error_handler

app.add_exception_handler(AppError, app_error_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
                "details": {"debug_message": str(exc)} if app.debug else {}
            }
        }
    )
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
collaborative_model = None
content_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global collaborative_model, content_model
    
    logger.info("Starting UMass Dining Recommender API...")
    
    try:
        # Initialize collaborative filtering model
        collaborative_model = DiningCollaborativeFilter()
        
        # Try to load pre-trained model
        model_files = list(Path("models").glob("collaborative_filter_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            collaborative_model.load_model(str(latest_model))
            logger.info(f"Loaded collaborative model: {latest_model}")
        else:
            logger.warning("No pre-trained collaborative model found")
        
        # Initialize content-based model
        content_model = ContentBasedRecommender()
        
        # Try to load pre-trained model
        content_files = list(Path("models").glob("content_based_*.pkl"))
        if content_files:
            latest_content = max(content_files, key=lambda x: x.stat().st_mtime)
            content_model.load_model(str(latest_content))
            logger.info(f"Loaded content model: {latest_content}")
        else:
            logger.warning("No pre-trained content model found")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

# Include routes
app.include_router(router, prefix="/api/v1")

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: str

class RecommendationResponse(BaseModel):
    user_id: str
    algorithm: str
    recommendations: list
    count: int
    timestamp: str

class RecommendationRequest(BaseModel):
    user_id: str
    dining_hall: Optional[str] = None
    meal_period: Optional[str] = None
    top_k: int = 10

@app.get("/")
async def root():
    """Root endpoint"""
    return StatusResponse(
        status="success",
        message="UMass Dining Recommender API",
        data={
            "version": "1.0.0",
            "endpoints": [
                "/api/v1/recommendations/{user_id}",
                "/api/v1/rate",
                "/api/v1/menu/{dining_hall}/{meal_period}",
                "/api/v1/user/{user_id}/stats",
                "/api/v1/dining-halls",
                "/api/v1/search",
                "/api/v1/health"
            ]
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/v1/health", response_model=StatusResponse)
def health_check():
    """Health check endpoint"""
    return StatusResponse(status="success", message="healthy", timestamp=datetime.now().isoformat())

@app.get("/api/v1/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    algorithm: str = "hybrid",
    top_k: int = 10,
    dining_hall: Optional[str] = None,
    meal_period: Optional[str] = None,
    include_rated: bool = False
):
    """
    Get personalized recommendations for a user.
    Hybrid logic: user ratings, dietary, meal period, popularity, diversity.
    Fallback to trending/popular if user is new.
    Explanations for each recommendation.
    """
    try:
        tracker = UserPreferenceTracker(user_id)
        try:
            user_context = tracker.get_recommendation_context()
        except Exception as context_error:
            logging.warning(f"Could not get user context: {context_error}")
            user_context = {}

        # Load all items
        items_path = Path("data/processed/unique_items.csv")
        if not items_path.exists():
            raise HTTPException(status_code=404, detail="No menu data available.")
        items_df = pd.read_csv(items_path)

        # Filter by meal period if provided
        if meal_period:
            meal_keywords = {
                "breakfast": ["egg", "pancake", "waffle", "bacon", "cereal", "oatmeal", "toast", "breakfast"],
                "lunch": ["sandwich", "burger", "salad", "wrap", "lunch", "chicken", "pizza"],
                "dinner": ["steak", "pasta", "dinner", "fish", "rice", "curry", "roast", "beef", "salmon"],
                "late night": ["pizza", "fries", "snack", "late night", "dessert"]
            }
            keywords = meal_keywords.get(meal_period.lower(), [])
            mask = items_df['item_name'].str.contains('|'.join(keywords), case=False, na=False)
            filtered_items = items_df[mask]
            if len(filtered_items) > 0:
                items_df = filtered_items

        # Filter by dietary restrictions
        dietary = set(user_context.get('dietary_restrictions', []))
        if dietary:
            if 'vegan' in dietary:
                items_df = items_df[items_df['is_vegan'] == True]
            elif 'vegetarian' in dietary:
                items_df = items_df[items_df['is_vegetarian'] == True]

        # Fallback: if no user ratings, recommend popular items
        rated_items = user_context.get('rated_items', set())
        if not rated_items:
            recs = items_df.nlargest(top_k, 'frequency')
            recommendations = []
            for _, row in recs.iterrows():
                recommendations.append({
                    "item_id": int(row['item_id']),
                    "item_name": row['item_name'],
                    "score": 0.7,
                    "confidence": 0.7,
                    "method": "popularity",
                    "station": row.get('station', ''),
                    "calories": row.get('calories'),
                    "protein": row.get('protein'),
                    "allergens": row.get('allergens', ''),
                    "is_vegan": row.get('is_vegan', False),
                    "is_vegetarian": row.get('is_vegetarian', False),
                    "explanations": ["Popular item", "No user ratings yet"]
                })
            return RecommendationResponse(
                user_id=user_id,
                algorithm=algorithm,
                recommendations=recommendations,
                count=len(recommendations),
                timestamp=datetime.now().isoformat()
            )

        # Hybrid: recommend items similar to high-rated, plus some diversity
        high_rated = set(user_context.get('high_rated_items', []))
        if high_rated:
            # Recommend similar items (by name/description)
            liked_items = items_df[items_df['item_id'].isin(high_rated)]
            liked_keywords = set()
            for name in liked_items['item_name']:
                liked_keywords.update(name.lower().split())
            mask = items_df['item_name'].apply(lambda x: any(word in x.lower() for word in liked_keywords))
            similar_items = items_df[mask & (~items_df['item_id'].isin(high_rated))]
            # Add some diversity
            diverse_items = items_df.sample(min(3, len(items_df)))
            recs = pd.concat([liked_items, similar_items, diverse_items]).drop_duplicates('item_id').head(top_k)
        else:
            recs = items_df.sample(min(top_k, len(items_df)))

        recommendations = []
        for _, row in recs.iterrows():
            explanations = []
            if row['item_id'] in high_rated:
                explanations.append("You rated this highly before")
            if row.get('is_vegan', False):
                explanations.append("Matches your vegan preference")
            if row.get('is_vegetarian', False):
                explanations.append("Vegetarian option")
            if meal_period:
                explanations.append(f"Recommended for {meal_period.title()}")
            if not explanations:
                explanations.append("Recommended for variety")
            recommendations.append({
                "item_id": int(row['item_id']),
                "item_name": row['item_name'],
                "score": 0.8,
                "confidence": 0.8,
                "method": "hybrid",
                "station": row.get('station', ''),
                "calories": row.get('calories'),
                "protein": row.get('protein'),
                "allergens": row.get('allergens', ''),
                "is_vegan": row.get('is_vegan', False),
                "is_vegetarian": row.get('is_vegetarian', False),
                "explanations": explanations
            })
        return RecommendationResponse(
            user_id=user_id,
            algorithm=algorithm,
            recommendations=recommendations,
            count=len(recommendations),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rate", response_model=StatusResponse)
def rate_item(request: Dict[str, Any]):
    """Rate a food item"""
    try:
        tracker = UserPreferenceTracker(request['user_id'])
        tracker.rate_item(
            item_id=request['item_id'],
            rating=request['rating'],
            item_name=request.get('item_name')
        )
        return StatusResponse(status="success", message=f"Rated {request.get('item_name') or request['item_id']}: {request['rating']}/5", timestamp=datetime.now().isoformat())
    except ValueError as e:
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())
    except Exception as e:
        logging.error(f"Error rating item: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.post("/api/v1/history", response_model=StatusResponse)
def add_to_history(request: Dict[str, Any]):
    """Record that user ate an item"""
    try:
        tracker = UserPreferenceTracker(request['user_id'])
        tracker.add_to_history(
            item_id=request['item_id'],
            dining_hall=request['dining_hall'],
            meal_period=request['meal_period'],
            item_name=request.get('item_name')
        )
        return StatusResponse(status="success", message="Added to eating history", timestamp=datetime.now().isoformat())
    except Exception as e:
        logging.error(f"Error adding to history: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.get("/api/v1/user/{user_id}/stats", response_model=StatusResponse)
def get_user_stats(user_id: str):
    """Get user statistics and preferences"""
    try:
        tracker = UserPreferenceTracker(user_id)
        stats = tracker.get_statistics()
        return StatusResponse(status="success", data={"user_id": user_id, "stats": stats}, timestamp=datetime.now().isoformat())
    except Exception as e:
        logging.error(f"Error getting user stats: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.get("/api/v1/items/{item_id}/similar")
async def get_similar_items(item_id: int, top_k: int = 10):
    """Get items similar to a given item"""
    try:
        if not content_model:
            raise HTTPException(status_code=503, detail="Content model not available")
        
        similar_items = content_model.get_similar_items(item_id, top_k)
        
        return {
            "item_id": item_id,
            "similar_items": similar_items,
            "count": len(similar_items),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting similar items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/items/{item_id}/profile")
async def get_item_profile(item_id: int):
    """Get detailed profile of an item"""
    try:
        if not content_model:
            raise HTTPException(status_code=503, detail="Content model not available")
        
        profile = content_model.get_item_profile(item_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Item not found")
        
        return {
            "item_id": item_id,
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting item profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search", response_model=StatusResponse)
def search_items(request: Dict[str, Any]):
    """Search for food items"""
    try:
        if not content_model:
            raise HTTPException(status_code=503, detail="Content model not available")
        
        results = content_model.find_items_by_criteria(
            request['criteria'],
            top_k=request['top_k']
        )
        
        return StatusResponse(status="success", data={"criteria": request['criteria'], "results": results, "count": len(results)}, timestamp=datetime.now().isoformat())
        
    except Exception as e:
        logging.error(f"Error searching items: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.post("/api/v1/dietary-preferences", response_model=StatusResponse)
def set_dietary_preferences(request: Dict[str, Any]):
    """Set user's dietary restrictions"""
    try:
        tracker = UserPreferenceTracker(request['user_id'])
        tracker.set_dietary_restrictions(request['restrictions'])
        return StatusResponse(status="success", message="Dietary preferences updated", data={"restrictions": request['restrictions']}, timestamp=datetime.now().isoformat())
    except Exception as e:
        logging.error(f"Error setting dietary preferences: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.get("/api/v1/dining-halls", response_model=StatusResponse)
def get_dining_halls():
    """Get list of all dining halls"""
    try:
        halls = [
            {"id": "worcester", "name": "Worcester Dining Commons", "location": "Southwest"},
            {"id": "franklin", "name": "Franklin Dining Commons", "location": "Southwest"},
            {"id": "berkshire", "name": "Berkshire Dining Commons", "location": "Central"},
            {"id": "hampshire", "name": "Hampshire Dining Commons", "location": "Central"}
        ]
        return StatusResponse(status="success", data={"dining_halls": halls}, timestamp=datetime.now().isoformat())
    except Exception as e:
        logging.error(f"Error getting dining halls: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.get("/api/v1/menu/{dining_hall}/{meal_period}", response_model=StatusResponse)
def get_menu(dining_hall: str, meal_period: str, date: Optional[str] = None):
    """Get menu for specific dining hall and meal period"""
    try:
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        menu_file = Path(f"data/raw/menus/menu_{date}.json")
        if not menu_file.exists():
            return StatusResponse(status="error", message="Menu not found for this date", timestamp=datetime.now().isoformat())
        import json
        with open(menu_file, 'r') as f:
            menu_data = json.load(f)
        if dining_hall not in menu_data['menus']:
            return StatusResponse(status="error", message=f"Dining hall '{dining_hall}' not found", timestamp=datetime.now().isoformat())
        hall_menu = menu_data['menus'][dining_hall]
        if meal_period not in hall_menu['meals']:
            return StatusResponse(status="error", message=f"Meal period '{meal_period}' not found", timestamp=datetime.now().isoformat())
        return StatusResponse(
            status="success",
            data={
                "dining_hall": dining_hall,
                "meal_period": meal_period,
                "date": menu_data['date'],
                "menu": hall_menu['meals'][meal_period]
            },
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Error getting menu: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.get("/api/v1/menu/today", response_model=StatusResponse)
def get_all_menus_today():
    """Get all dining hall menus for today"""
    try:
        date = datetime.now().strftime('%Y%m%d')
        menu_file = Path(f"data/raw/menus/menu_{date}.json")
        if not menu_file.exists():
            return StatusResponse(status="error", message="Today's menu not available yet", timestamp=datetime.now().isoformat())
        import json
        with open(menu_file, 'r') as f:
            menu_data = json.load(f)
        return StatusResponse(
            status="success",
            data={
                "date": menu_data['date'],
                "menus": menu_data['menus']
            },
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Error getting all menus: {e}")
        return StatusResponse(status="error", message=str(e), timestamp=datetime.now().isoformat())

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return StatusResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
