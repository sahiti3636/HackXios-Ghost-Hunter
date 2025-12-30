#!/bin/bash

# Ghost Hunter Setup & Run Script
# Usage: ./setup.sh

set -e  # Exit on error

echo "👻 Starting Ghost Hunter Setup..."

# 1. Setup Python Virtual Environment
echo "📦 Checking Python environment..."
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "⬇️  Installing backend dependencies..."
pip install --upgrade pip
if [ -f "requirements_backend.txt" ]; then
    pip install -r requirements_backend.txt
else
    pip install -r requirements.txt
fi

# 2. Setup Frontend Dependencies
echo "💻 Checking Frontend dependencies..."
cd ghost-hunter-frontend

if [ ! -d "node_modules" ]; then
    echo "   Installing node modules (this may take a moment)..."
    npm install
fi

cd ..

# 3. Launch Application
echo "🚀 Launching Ghost Hunter..."
echo "   - Backend: http://localhost:5001"
echo "   - Frontend: http://localhost:3000"
echo "👉 Press Ctrl+C to stop both servers."

# Trap SIGINT (Ctrl+C) to kill all child processes
trap 'kill 0' SIGINT

# Start Backend in background
echo "🔥 Starting Backend..."
python app.py &

# Start Frontend in background
echo "✨ Starting Frontend..."
cd ghost-hunter-frontend
npm run dev &

# Wait for all background processes to finish (keeps script running)
wait
