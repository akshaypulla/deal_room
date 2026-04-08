#!/bin/bash
# DealRoom Hugging Face Spaces Deployment Script

set -e

echo "=== DealRoom Hugging Face Spaces Deployment ==="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists openenv; then
    echo "ERROR: openenv CLI not found. Install with: pip install openenv-core[core]"
    exit 1
fi

if ! command_exists docker; then
    echo "WARNING: docker not found. Local build won't work."
fi

if ! command_exists git; then
    echo "ERROR: git is required for Hugging Face Spaces deployment."
    exit 1
fi

echo "Prerequisites check passed."
echo ""

# Validate environment
echo "Validating OpenEnv environment..."
openenv validate
echo ""

# Get repository ID from user or use default
REPO_ID="${1:-}"

if [ -z "$REPO_ID" ]; then
    echo "No repository ID provided."
    echo "Usage: ./deploy.sh <username/deal-room>"
    echo "   or: HF_USERNAME=<username> ./deploy.sh"
    echo ""
    
    # Try to get from openenv.yaml or use default
    if [ -f "openenv.yaml" ]; then
        AUTHOR=$(grep "author:" openenv.yaml | cut -d'"' -f2)
        NAME=$(grep "^name:" openenv.yaml | cut -d'"' -f2)
        if [ -n "$AUTHOR" ] && [ -n "$NAME" ]; then
            REPO_ID="${AUTHOR}/${NAME}"
        fi
    fi
    
    if [ -z "$REPO_ID" ]; then
        REPO_ID="username/deal-room"
    fi
    
    echo "Using default repository ID: $REPO_ID"
    echo "To specify a different ID, provide it as argument: ./deploy.sh <repo-id>"
    echo ""
fi

# Build the environment
echo "Building Docker image..."
openenv build
echo ""

# Push to Hugging Face Spaces
echo "Pushing to Hugging Face Spaces..."
echo "Repository: $REPO_ID"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

openenv push -r "$REPO_ID" --interface

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Your DealRoom environment is now live at:"
echo "  https://huggingface.co/spaces/$REPO_ID"
echo ""
echo "API Endpoints:"
echo "  - /health - Health check"
echo "  - /metadata - Environment metadata"
echo "  - /reset - Reset environment"
echo "  - /step - Execute action"
echo "  - /state - Get current state"
echo "  - /web - Web interface"
