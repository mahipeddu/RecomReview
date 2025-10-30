#!/bin/bash

# Quick Start Script for Streamlit Reviewer Recommendation App
# This script starts the Streamlit web application

echo "=================================================="
echo "   Reviewer Recommendation System - Web App"
echo "=================================================="
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly --quiet
    echo "✅ Installation complete!"
    echo ""
fi

# Check if embeddings cache exists
if [ ! -f "pretrained_embedding/embeddings_cache.pt" ]; then
    echo "❌ WARNING: Embeddings cache not found!"
    echo "   File missing: pretrained_embedding/embeddings_cache.pt"
    echo ""
    echo "Please generate embeddings first:"
    echo "   cd pretrained_embedding"
    echo "   python3 semantic_similarity_gpu.py"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will be available at:"
echo "   Local:    http://localhost:8501"
echo "   Network:  Check the output below"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Start Streamlit
python3 -m streamlit run streamlit_app.py --server.headless true --server.port 8501
