# UMass Dining Recommendation Engine

# UMass Dining Recommender System

A full-stack application that provides personalized dining recommendations for UMass students.

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm 9 or higher
- pip (Python package installer)

## Project Structure

```
umass-dining/
├── umass-dining-frontend/     # Next.js frontend application
└── umass-dining-recommender/  # FastAPI backend application
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd umass-dining
```

2. Backend Setup:
```bash
cd umass-dining-recommender
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Frontend Setup:
```bash
cd ../umass-dining-frontend
npm install
```

4. Environment Configuration:
   - Create `.env.local` in the frontend directory:
     ```
     NEXT_PUBLIC_API_URL=http://localhost:8001/api/v1
     ```

## Running the Application

1. Start the Backend Server:
```bash
cd umass-dining-recommender
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
python -m uvicorn api.main:app --reload --port 8001
```

2. Start the Frontend Development Server (in a new terminal):
```bash
cd umass-dining-frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

A complete AI-powered dining recommendation system for UMass students, featuring web scraping, machine learning, and a modern web interface.

## 🚀 Quick Start

### Option 1: Complete System (Recommended)
```bash
./start_complete_system.sh
```

### Option 2: Individual Components
```bash
# Terminal 1 - Backend API
./start_backend.sh

# Terminal 2 - Frontend
./start_frontend.sh
```

## 📁 Project Structure

```
umass-dining/
├── umass-dining-recommender/     # Backend API (Python/FastAPI)
│   ├── scrapers/                 # Web scraping modules
│   ├── data_processing/          # Data cleaning & NLP
│   ├── models/                   # ML recommendation models
│   ├── api/                      # FastAPI application
│   └── data/                     # Data storage
├── umass-dining-frontend/        # Frontend (Next.js/React)
│   ├── src/app/                  # Next.js pages
│   ├── src/components/           # React components
│   └── public/                   # Static assets
└── scripts/                      # Utility scripts
```

## 🛠️ Features

### Backend (Python/FastAPI)
- **Web Scraping**: Automated menu data collection
- **Data Processing**: Cleaning and standardization
- **NLP Embeddings**: Semantic food item understanding
- **ML Models**: Collaborative filtering, content-based, hybrid
- **REST API**: Comprehensive API endpoints
- **User Tracking**: Preference and rating management

### Frontend (Next.js/React)
- **Modern UI**: Responsive design with Tailwind CSS
- **Real-time Updates**: Live recommendation updates
- **User Interaction**: Rating and preference management
- **Search & Filter**: Advanced filtering options
- **Statistics**: User analytics and insights

## 🔧 Development

### Backend Development
```bash
cd umass-dining-recommender
source venv/bin/activate
python run.py api
```

### Frontend Development
```bash
cd umass-dining-frontend
npm run dev
```

### Testing
```bash
cd umass-dining-recommender
python test_system.py
```

## 📊 API Endpoints

- `GET /api/v1/recommendations/{user_id}` - Get recommendations
- `POST /api/v1/rate` - Rate a food item
- `GET /api/v1/dining-halls` - List dining halls
- `GET /api/v1/search` - Search items
- `GET /api/v1/user/{user_id}/stats` - User statistics
- `GET /docs` - Interactive API documentation

## 🎯 Usage

1. **Start the system** using the startup scripts
2. **Visit the frontend** at http://localhost:3000
3. **Enter your user ID** (e.g., "justin_manoj")
4. **Rate some items** to build your preference profile
5. **Get personalized recommendations** based on your preferences

## 📚 Documentation

- [Backend API Documentation](umass-dining-recommender/README.md)
- [Frontend Documentation](umass-dining-frontend/README.md)
- [API Reference](http://localhost:8000/docs) (when running)

ALL RIGHTS RESERVED 
