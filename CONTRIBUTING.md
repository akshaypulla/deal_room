# Contributing to DealRoom

Thank you for your interest in contributing to DealRoom!

## Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd deal_room

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -q

# Validate OpenEnv spec
openenv validate
```

## Code Standards

- All code must pass pytest tests
- Follow existing patterns in the codebase
- No LLM calls in the `deal_room/` package (deterministic only)
- Update documentation if you change functionality

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally
5. Submit a pull request

## Reporting Issues

Please report issues on the GitHub issue tracker with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior