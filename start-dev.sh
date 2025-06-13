#!/bin/bash

# Mock Data Generator - Full Stack Development Startup Script
# This script starts the Model Server, Backend, and Frontend in parallel.

echo "🚀 Starting Mock Data Generator Development Environment..."
echo "======================================================"

# Function to kill all background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down all services..."
    # The 'jobs -p' command gets the PIDs of all background jobs started in this script.
    # The 'kill' command sends a termination signal to them.
    kill $(jobs -p) 2>/dev/null
    echo "✅ All services stopped."
    exit 0
}

# Trap signals (like Ctrl+C) and call the cleanup function
trap cleanup SIGINT SIGTERM

# --- Step 1: Check Prerequisites ---
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment '.venv' not found. Please run the setup steps in the README.md first."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "❌ Environment file '.env' not found. Please create it and add your GGUF_MODEL_PATH."
    exit 1
fi

if ! redis-cli ping >/dev/null 2>&1; then
    echo "❌ Redis is not running or not accessible."
    echo "Please start Redis using 'brew services start redis' or another method."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include source directories
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src:$(pwd)/model_server/src

# --- Step 2: Start Model Server ---
echo "🧠 Starting Model Server on port 8001..."
python -m uvicorn model_server.src.main:app --host 0.0.0.0 --port 8001 &
MODEL_SERVER_PID=$!
sleep 5 # Give it a moment to load models

# --- Step 3: Start Backend ---
echo "⚙️ Starting Backend Server on port 8000..."
python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
sleep 3

# --- Step 4: Start Frontend ---
echo "🎨 Starting Frontend (React) on port 3000..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# --- All Done ---
echo ""
echo "✅ All services are starting up!"
echo "-----------------------------------"
echo "🟢 Frontend App:    http://localhost:3000"
echo "🟢 Backend API:     http://localhost:8000/api/docs"
echo "🟢 Model Server API:  http://localhost:8001/docs"
echo "-----------------------------------"
echo ""
echo "Press Ctrl+C in this terminal to shut down all services gracefully."
echo ""

# Wait for any of the background processes to exit.
# The 'wait' command without a PID waits for all child processes.
# The 'cleanup' function will handle shutting everything down.
wait 