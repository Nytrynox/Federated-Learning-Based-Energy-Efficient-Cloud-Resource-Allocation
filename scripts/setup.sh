#!/bin/bash

# Federated Learning Energy-Efficient Cloud Resource Allocation
# Quick Start Script

set -e

echo "🚀 Starting Federated Learning Energy-Efficient Cloud Resource Allocation System"
echo "=================================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data models config/secrets

# Set environment variables
export PYTHONPATH=$(pwd)/src
export FLASK_ENV=development

echo ""
echo "🎯 Setup complete! Here's what you can do next:"
echo ""
echo "1. 🧪 Run simulation:"
echo "   python src/simulation/run_simulation.py"
echo ""
echo "2. 🌐 Start API server:"
echo "   python src/api/app.py"
echo ""
echo "3. 🐳 Run with Docker:"
echo "   docker-compose up"
echo ""
echo "4. ☁️  Deploy to Azure:"
echo "   azd up"
echo ""
echo "📖 For more information, see README.md"
echo ""

# Offer to run simulation
read -p "🤔 Would you like to run a quick simulation now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃‍♂️ Running simulation..."
    python src/simulation/run_simulation.py
else
    echo "👋 Setup complete! Run any of the commands above to get started."
fi
