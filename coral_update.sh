#!/bin/bash

# Coral Update Script
# Pull latest changes from git repository

echo "🚀 Updating visuomotor-deep-optics on Coral..."

# Navigate to project directory
cd ~/visuomotor-deep-optics/

# Check current status
echo "📍 Current status:"
git status --porcelain

# Pull latest changes
echo "⬇️  Pulling latest changes from dev branch..."
git pull origin dev

# Show what changed
echo "📋 Recent commits:"
git log --oneline -5

echo "✅ Update complete!"
echo ""
echo "🔧 Available SML tools:"
echo "  - Dataset generation: optomech/supervised_ml/sml_job_manager_simple.py"
echo "  - Training: optomech/supervised_ml/train_sml_model.py"
echo "  - Dataset checkout: optomech/supervised_ml/dataset_checkout.py"
