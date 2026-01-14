# Contributing to ADAPTA

First off, thank you for considering contributing to ADAPTA! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if possible**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow our code style guidelines
- Include appropriate test cases
- Update documentation as needed
- Ensure the test suite passes
- Make sure your code lints

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR-USERNAME/brfss-heart-disease-ml.git
cd brfss-heart-disease-ml
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install black flake8 pytest  # Development tools
```

4. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

5. Make your changes and test:
```bash
python test_performance.py
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update documentation in `docs/` if you changed functionality
3. The PR will be merged once you have approval from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added that prove fix/feature works
- [ ] All tests pass locally

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use `black` for formatting:

```bash
# Format code
black src/

# Check linting
flake8 src/
```

### Code Standards

- **Docstrings**: All functions/classes must have docstrings
- **Type Hints**: Use type hints where applicable
- **Naming**: Use descriptive variable names
- **Functions**: Keep functions focused and small
- **Comments**: Explain complex logic

### Example

```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    # Implementation here
    pass
```

## Testing

All new features should include tests:

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_models.py
```

## Documentation

Update documentation when:
- Adding new features
- Changing existing functionality
- Fixing significant bugs

Documentation is in `docs/` directory:
- `methodology.md`: Technical details
- `api.md`: API reference
- `quickstart.md`: Getting started guide

## Questions?

Feel free to open an issue with your question or contact the maintainers:
- Herald M.S. Theo
- Fera C.W. Hamid

---

Thank you for contributing to ADAPTA! 🚀
