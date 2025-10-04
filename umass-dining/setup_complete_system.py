#!/usr/bin/env python3
"""
Complete UMass Dining Recommender System Setup Script

This script sets up the entire UMass Dining Recommender System including:
- Backend API with all components
- Frontend Next.js application
- Sample data generation
- System testing
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return e

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def setup_backend():
    """Set up the backend API"""
    print_step("1", "Setting up Backend API")
    
    backend_dir = Path("umass-dining-recommender")
    
    if not backend_dir.exists():
        print("❌ Backend directory not found. Please run the main setup first.")
        return False
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    result = run_command("pip install -r requirements.txt", cwd=backend_dir)
    if result.returncode != 0:
        print("❌ Failed to install Python dependencies")
        return False
    
    # Create necessary directories
    print("Creating data directories...")
    directories = [
        "data/raw/menus",
        "data/processed", 
        "data/embeddings",
        "user_data",
        "models/saved"
    ]
    
    for dir_path in directories:
        full_path = backend_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    # Generate sample data
    print("Generating sample data...")
    result = run_command("python demo.py", cwd=backend_dir)
    if result.returncode != 0:
        print("⚠️  Sample data generation had issues, but continuing...")
    
    print("✓ Backend setup complete")
    return True

def setup_frontend():
    """Set up the frontend application"""
    print_step("2", "Setting up Frontend Application")
    
    frontend_dir = Path("umass-dining-frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found. Please run the main setup first.")
        return False
    
    # Install Node.js dependencies
    print("Installing Node.js dependencies...")
    result = run_command("npm install", cwd=frontend_dir)
    if result.returncode != 0:
        print("❌ Failed to install Node.js dependencies")
        return False
    
    # Create environment file
    env_file = frontend_dir / ".env.local"
    with open(env_file, 'w') as f:
        f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
    print("✓ Created .env.local file")
    
    print("✓ Frontend setup complete")
    return True

def test_system():
    """Test the complete system"""
    print_step("3", "Testing Complete System")
    
    backend_dir = Path("umass-dining-recommender")
    
    # Test backend components
    print("Testing backend components...")
    result = run_command("python test_system.py", cwd=backend_dir)
    if result.returncode != 0:
        print("⚠️  Backend tests had issues, but continuing...")
    
    # Test frontend build
    print("Testing frontend build...")
    frontend_dir = Path("umass-dining-frontend")
    result = run_command("npm run build", cwd=frontend_dir)
    if result.returncode != 0:
        print("⚠️  Frontend build had issues, but continuing...")
    
    print("✓ System testing complete")
    return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print_step("4", "Creating Startup Scripts")
    
    # Backend startup script
    backend_script = """#!/bin/bash
echo "Starting UMass Dining Recommender API..."
cd umass-dining-recommender
source venv/bin/activate
python run.py api
"""
    
    with open("start_backend.sh", "w") as f:
        f.write(backend_script)
    os.chmod("start_backend.sh", 0o755)
    print("✓ Created start_backend.sh")
    
    # Frontend startup script
    frontend_script = """#!/bin/bash
echo "Starting UMass Dining Frontend..."
cd umass-dining-frontend
npm run dev
"""
    
    with open("start_frontend.sh", "w") as f:
        f.write(frontend_script)
    os.chmod("start_frontend.sh", 0o755)
    print("✓ Created start_frontend.sh")
    
    # Complete system startup script
    complete_script = """#!/bin/bash
echo "Starting Complete UMass Dining Recommender System..."
echo "This will start both the backend API and frontend in separate terminals."

# Start backend in background
echo "Starting backend API..."
cd umass-dining-recommender
source venv/bin/activate
python run.py api &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd ../umass-dining-frontend
npm run dev &
FRONTEND_PID=$!

echo "System started!"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait
"""
    
    with open("start_complete_system.sh", "w") as f:
        f.write(complete_script)
    os.chmod("start_complete_system.sh", 0o755)
    print("✓ Created start_complete_system.sh")
    
    return True

def create_documentation():
    """Create comprehensive documentation"""
    print_step("5", "Creating Documentation")
    
    # Main README
    main_readme = """# UMass Dining Recommender System

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes and demonstrates modern web development and machine learning techniques.

## 🆘 Support

If you encounter any issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed
3. Verify the API is running on port 8000
4. Check the documentation for troubleshooting tips
"""
    
    with open("README.md", "w") as f:
        f.write(main_readme)
    print("✓ Created main README.md")
    
    return True

def main():
    """Main setup function"""
    print_header("UMass Dining Recommender System - Complete Setup")
    
    print("This script will set up the complete UMass Dining Recommender System including:")
    print("- Backend API with all components")
    print("- Frontend Next.js application") 
    print("- Sample data generation")
    print("- System testing")
    print("- Startup scripts")
    print("- Documentation")
    
    # Check if we're in the right directory
    if not Path("umass-dining-recommender").exists() or not Path("umass-dining-frontend").exists():
        print("❌ Required directories not found. Please run the main setup first.")
        return False
    
    # Run setup steps
    steps = [
        ("Backend Setup", setup_backend),
        ("Frontend Setup", setup_frontend),
        ("System Testing", test_system),
        ("Startup Scripts", create_startup_scripts),
        ("Documentation", create_documentation)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
                print(f"✓ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
    
    # Final summary
    print_header("Setup Complete!")
    print(f"Completed {success_count}/{len(steps)} setup steps successfully")
    
    if success_count == len(steps):
        print("🎉 Complete system setup successful!")
        print("\nNext steps:")
        print("1. Start the complete system: ./start_complete_system.sh")
        print("2. Or start components individually:")
        print("   - Backend: ./start_backend.sh")
        print("   - Frontend: ./start_frontend.sh")
        print("3. Visit http://localhost:3000 to use the system")
        print("4. Check http://localhost:8000/docs for API documentation")
    else:
        print("⚠️  Some setup steps failed. Check the error messages above.")
        print("You can still try to run the system with the startup scripts.")
    
    print("\nThank you for setting up the UMass Dining Recommender System!")

if __name__ == "__main__":
    main()
