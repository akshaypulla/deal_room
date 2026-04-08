# Makefile for DealRoom Hugging Face Spaces Deployment

.PHONY: build push validate test docker-build docker-run

# Default target
all: validate build

# Validate the OpenEnv environment
validate:
	openenv validate

# Build for Hugging Face Spaces
build:
	openenv build

# Push to Hugging Face Spaces
push:
	openenv push

# Run tests
test:
	python -m pytest tests/ -v

# Build Docker image locally
docker-build:
	docker build -f Dockerfile.huggingface -t deal-room:latest .

# Run Docker container locally
docker-run:
	docker run --rm -p 7860:7860 --name deal-room deal-room:latest

# Clean up
clean:
	docker rmi deal-room:latest 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies
install:
	pip install -r requirements.txt
	pip install gradio>=4.0.0
	pip install openenv-core[core]>=0.2.2

# Run the server locally
run:
	uvicorn server.app:app --reload --port 7860

# Format code
format:
	python -m ruff format .

# Lint code
lint:
	python -m ruff check .
