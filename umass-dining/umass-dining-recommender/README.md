# UMass Dining Recommender

An AI-powered dining recommendation system for UMass students, featuring collaborative filtering, content-based recommendations, and hybrid approaches.

## 🚀 Features

- **Smart Recommendations**: Get personalized food recommendations based on your preferences and dining history
- **Multiple Algorithms**: Collaborative filtering, content-based, and hybrid recommendation approaches
- **Real-time Menu Scraping**: Automatically scrapes daily menus from all UMass dining halls
- **Nutritional Analysis**: Comprehensive nutritional information and health scoring
- **User Preference Tracking**: Track your ratings, dietary restrictions, and eating patterns
- **RESTful API**: Complete API for integration with mobile apps and web interfaces
- **Analytics Dashboard**: Insights into dining patterns and recommendation performance


```
umass-dining-recommender/
├── scrapers/              # Data collection modules
│   ├── menu_scraper.py    # Scrape daily menus
│   ├── nutrition_scraper.py # Get nutritional data
│   └── scheduler.py       # Automated scraping
├── data_processing/       # Data cleaning and preparation
│   ├── clean_menus.py     # Clean and standardize data
│   ├── item_embeddings.py # NLP embeddings for food
│   └── feature_engineering.py # Create features
├── models/                # Recommendation models
│   ├── collaborative_filter.py # Collaborative filtering
│   ├── content_based.py   # Content-based recommendations
│   ├── hybrid_model.py    # Hybrid approach
│   └── user_preferences.py # User preference tracking
├── api/                   # FastAPI application
│   ├── main.py           # Main API app
│   ├── routes.py         # API endpoints
│   └── models.py         # Pydantic models
├── data/                  # Data storage
│   ├── raw/              # Scraped data
│   ├── processed/        # Cleaned data
│   └── embeddings/       # Food embeddings
└── config/               # Configuration files
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip
- Redis (optional, for caching)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd umass-dining-recommender
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize data directories**
   ```bash
   mkdir -p data/{raw,processed,embeddings} logs models user_data
   ```

## 🚀 Quick Start

### 1. Start the API Server

```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

### 2. Scrape Menu Data

```bash
# Scrape today's menus
python scrapers/menu_scraper.py

# Run automated scheduler
python scrapers/scheduler.py --start
```

### 3. Train Recommendation Models

```bash
# Train collaborative filtering model
python models/collaborative_filter.py

# Train content-based model
python models/content_based.py
```

### 4. Get Recommendations

```bash
# Using the API
curl -X GET "http://localhost:8000/api/v1/recommendations/user123?algorithm=hybrid&top_k=10"

# Using Python
python -c "
from models.user_preferences import UserPreferenceTracker
tracker = UserPreferenceTracker('user123')
tracker.rate_item(1, 5, 'Chicken Tikka Masala')
print('User preferences updated!')
"
```

## 📊 API Documentation

### Core Endpoints

- `GET /api/v1/recommendations/{user_id}` - Get recommendations
- `POST /api/v1/rate` - Rate a food item
- `POST /api/v1/history` - Add to eating history
- `GET /api/v1/user/{user_id}/stats` - Get user statistics
- `GET /api/v1/items/{item_id}/similar` - Get similar items
- `GET /api/v1/menu/{dining_hall}` - Get dining hall menu

### Example API Usage

```python
import requests

# Rate an item
response = requests.post("http://localhost:8000/api/v1/rate", json={
    "user_id": "user123",
    "item_id": 1,
    "rating": 5,
    "item_name": "Chicken Tikka Masala",
    "dining_hall": "worcester"
})

# Get recommendations
response = requests.get("http://localhost:8000/api/v1/recommendations/user123")
recommendations = response.json()["recommendations"]
```

## 🧠 Recommendation Algorithms

### 1. Collaborative Filtering
- Uses matrix factorization to find users with similar preferences
- Learns from user ratings and interactions
- Good for finding popular items among similar users

### 2. Content-Based Filtering
- Analyzes item features (nutrition, station, cuisine type)
- Uses TF-IDF and cosine similarity
- Good for finding items similar to liked items

### 3. Hybrid Approach
- Combines collaborative and content-based methods
- Weighted combination of both approaches
- Provides most accurate recommendations

## 📈 Data Pipeline

1. **Data Collection**: Automated scraping of daily menus
2. **Data Cleaning**: Standardization and normalization
3. **Feature Engineering**: Create features for ML models
4. **Model Training**: Train recommendation models
5. **Serving**: Real-time recommendation API

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
COLLABORATIVE_FACTORS=50
CONTENT_SIMILARITY_THRESHOLD=0.3

# Scraping Configuration
SCRAPING_ENABLED=True
SCRAPING_INTERVAL=3600
```

### Model Parameters

- **Collaborative Filtering**: `n_factors=50`, `learning_rate=0.01`
- **Content-Based**: `similarity_threshold=0.3`, `max_features=1000`
- **Hybrid**: `collaborative_weight=0.6`, `content_weight=0.4`

## 📊 Monitoring and Analytics

### Logging
- Structured logging with different levels
- Log rotation and archival
- Performance metrics tracking

### Analytics
- User engagement metrics
- Recommendation accuracy
- Model performance monitoring

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api/main.py"]
```

### Production Considerations

- Use Redis for caching
- Set up proper logging
- Configure monitoring
- Use environment variables for secrets
- Set up automated backups


## 🙏 Acknowledgments

- UMass Dining Services for menu data
- FastAPI for the web framework
- PyTorch for machine learning
- scikit-learn for additional ML tools

