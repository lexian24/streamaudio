#!/bin/bash
echo "⚡ Starting FastAudio Application..."

# Start backend in background
echo "Starting backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend if available
if [ -f "start_frontend.sh" ]; then
    echo "Starting frontend..."
    ./start_frontend.sh &
    FRONTEND_PID=$!
    
    echo ""
    echo "✅ FastAudio started!"
    echo "📱 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "🏥 Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop both services"
    
    # Wait for user interrupt
    wait
else
    echo ""
    echo "✅ Backend started!"
    echo "🔧 Backend API: http://localhost:8000"
    echo "🏥 Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop"
    
    # Wait for backend
    wait $BACKEND_PID
fi