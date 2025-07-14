s# Quick Reference for AI Assistants

## 🚀 Essential Commands

### Project Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "import adaptive_bayesian_driver; print('✓ Package ready')"
```

### Testing & Validation
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_environment.py -v

# Check formatting issues
python _chatbot/_diagnostics/format_scanner.py

# Quick CI/CD check
python _chatbot/_diagnostics/cicd_check.py

# Comprehensive validation
python _chatbot/_diagnostics/validate_build.py
```

### Fixing Common Issues
```bash
# Fix formatting problems
python _chatbot/_fixing/fix_formatting.py

# Clean whitespace
python _chatbot/_fixing/clean_whitespace.py

# Check for linting errors
python -m flake8 adaptive_bayesian_driver/ --count
```

### Git Workflow
```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to remote
git push origin branch-name
```

## 📁 Directory Structure

```
├── adaptive_bayesian_driver/    # Main package
│   ├── models/                  # Learning algorithms
│   ├── environment/             # Simulation components
│   ├── utils/                   # Utilities
│   └── applications/            # Domain applications
├── tests/                       # Test suites
├── config/                      # Configuration files
├── notebooks/                   # Jupyter notebooks
├── _chatbot/                    # AI assistant tools
│   ├── _diagnostics/           # Diagnostic scripts
│   ├── _fixing/                # Repair utilities
│   └── _testing/               # Test helpers
├── _reports/                    # Generated reports
└── _assistant/                  # Session guidelines
```

## 🔧 Common File Operations

### Creating Helper Scripts
```bash
# Diagnostic tools go here:
_chatbot/_diagnostics/my_scanner.py

# Fixing tools go here:
_chatbot/_fixing/my_fix.py

# Never create scripts in project root!
```

### Importing the Package
```python
# Main package
import adaptive_bayesian_driver

# Specific components
from adaptive_bayesian_driver.models import RecursiveBayesianLearner
from adaptive_bayesian_driver.environment import SceneRenderer
from adaptive_bayesian_driver.config import load_config
```

### Running Experiments
```python
# Load configuration
from adaptive_bayesian_driver.config import load_config
config = load_config('config/experiment.yaml')

# Initialize components
from adaptive_bayesian_driver.models import RecursiveBayesianLearner
learner = RecursiveBayesianLearner(config)

# Run experiment
results = learner.train(data)
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Check package structure
python -c "import adaptive_bayesian_driver; print('OK')"

# Reinstall package
pip install -e .
```

### Test Failures
```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_specific.py::TestClass::test_method -v
```

### Formatting Issues
```bash
# Scan for problems
python _chatbot/_diagnostics/format_scanner.py

# Auto-fix most issues
python _chatbot/_fixing/fix_formatting.py
```

### CI/CD Failures
```bash
# Quick health check
python _chatbot/_diagnostics/cicd_check.py

# Check linting
python -m flake8 adaptive_bayesian_driver/ --count

# Verify package builds
python setup.py check
```

## 📝 Code Snippets

### Basic Test Template
```python
import unittest
import numpy as np
from adaptive_bayesian_driver.models import YourModel

class TestYourModel(unittest.TestCase):
    def setUp(self):
        self.model = YourModel()

    def test_basic_functionality(self):
        result = self.model.process(test_data)
        self.assertIsNotNone(result)
```

### Configuration Loading
```python
from adaptive_bayesian_driver.config import load_config

# Load default config
config = load_config()

# Load specific config
config = load_config('config/custom.yaml')
```

### Error Handling Template
```python
try:
    # Your code here
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
```

## ⚡ Performance Tips

### Memory Management
```python
# Use generators for large datasets
def process_data_generator(data):
    for item in data:
        yield process_item(item)

# Clear variables when done
del large_variable
```

### Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
logger.debug(f"Variable value: {variable}")
```

## 🔍 Useful Commands

### System Information
```bash
# Python version
python --version

# Package versions
pip list | grep torch
pip list | grep numpy

# Environment info
python -c "import sys; print(sys.path)"
```

### File Operations
```bash
# Find files
find . -name "*.py" -type f

# Search in files
grep -r "pattern" adaptive_bayesian_driver/

# Count lines of code
find . -name "*.py" -exec wc -l {} +
```

---

*Keep this reference handy for quick access to common operations and troubleshooting steps.*
