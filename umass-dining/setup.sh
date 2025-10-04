#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🚀 Setting up UMass Dining Recommender System..."

# Check for required tools
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is required but not installed."
    exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
echo "🔄 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd umass-dining-recommender
pip install -r requirements.txt

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd ../umass-dining-frontend
npm install

# Create a new tmux session for running both servers
if command_exists tmux; then
    echo "🚀 Starting servers in tmux session..."
    tmux new-session -d -s dining_app
    
    # Start backend server
    tmux send-keys -t dining_app 'cd ../umass-dining-recommender && source venv/bin/activate && python -m uvicorn api.main:app --reload --port 8001' C-m
    
    # Split window and start frontend server
    tmux split-window -h
    tmux send-keys -t dining_app 'cd ../umass-dining-frontend && npm run dev' C-m
    
    # Attach to the session
    tmux attach -t dining_app
else
    echo "⚠️ tmux not found. Starting servers in separate terminals..."
    echo "Please run these commands in separate terminals:"
    echo ""
    echo "Terminal 1 (Backend):"
    echo "cd umass-dining-recommender && source venv/bin/activate && python -m uvicorn api.main:app --reload --port 8001"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "cd umass-dining-frontend && npm run dev"
fi