#!/bin/bash
echo "⚡ Starting FastAudio Frontend..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start the React development server
npm start