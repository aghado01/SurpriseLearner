# Adaptive Bayesian Driver

A professional, biologically-inspired framework for adaptive learning and decision-making in non-stationary environments. This project implements computational models inspired by the locus coeruleus-norepinephrine (LC-NE) system for adaptive visual decision making with applications in autonomous driving and medical imaging.

## 🎯 Overview

This project implements a computational framework that demonstrates adaptive decision-making in non-stationary visual environments using recursive Bayesian updating, surprise detection, and generative priors. The system learns to navigate complex scenarios while adapting to:

- **Time-varying environmental statistics** (changing traffic patterns, evolving medical conditions)
- **Context-dependent rules** (danger sign avoidance, anomaly detection)
- **Perceptual uncertainty** (varying visual difficulty, noise levels)
- **Multi-modal sensory inputs** (visual, temporal, spatial features)

### Key Features

- **Recursive Bayesian Learning**: Dual-mode inference (particle filter + variational) for robust uncertainty estimation
- **Surprise Detection**: Multi-modal surprise metrics with adaptive thresholding
- **Volatility Control**: HMM-based volatility modeling for non-stationary environments
- **Scene Rendering**: Configurable environments for driving and medical imaging scenarios
- **Medical Applications**: Longitudinal MRI processing for anomaly detection
- **Professional Engineering**: Comprehensive testing, CI/CD, and configuration management

## 📁 Project Structure

```
surprise-learning-exploration-exploitation-wip/
├── adaptive_bayesian_driver/     # Main package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── main.py                  # CLI entry point
│   ├── models/                  # Core learning models
│   │   ├── base.py              # Base classes and generative models
│   │   ├── recursive_bayesian.py # Recursive Bayesian learner
│   │   ├── surprise.py          # Surprise detection and metrics
│   │   └── utils_particle.py    # Particle filter utilities
│   ├── environment/             # Environment simulation
│   │   ├── volatility.py        # Volatility controllers
│   │   ├── scene_renderer.py    # Scene rendering (driving/MNIST)
│   │   └── task_hmm.py          # Hidden Markov Models
│   ├── utils/                   # Utility functions
│   │   └── visualization.py     # Plotting and analysis
│   └── applications/            # Domain-specific applications
│       └── medical_imaging.py   # MRI anomaly detection
├── config/                      # Configuration files
│   └── experiment.yaml          # Experiment configurations
├── tests/                       # Test suites
│   ├── test_environment.py      # Environment tests
│   └── test_recursive_bayes.py  # Model tests
├── .github/workflows/           # CI/CD pipeline
│   └── ci.yml                   # GitHub Actions workflow
├── _assistant/                  # AI assistant guidelines
│   ├── README.md               # Assistant guidelines overview
│   ├── SESSION_GUIDELINES.md   # Session management procedures
│   ├── CONTEXT_SHARING.md      # Context sharing standards
│   ├── QUICK_COMMANDS.md       # Command reference for assistants
│   └── templates/              # Session and handoff templates
├── _reports/                   # Development reports
│   ├── README.md               # Report generation standards
│   ├── generate-report.ps1     # Automated report generation
│   └── templates/              # Report templates
├── notebooks/                   # Jupyter notebooks
│   └── demo.ipynb              # Demonstration notebook
├── deprecated/                  # Legacy code (archived)
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── REPO_GUIDELINES.md          # Development standards
├── QUICK_REFERENCE.md          # Quick command reference
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/surprise-learning-exploration-exploitation-wip.git
cd surprise-learning-exploration-exploitation-wip

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
# Run a complete experiment
python -m adaptive_bayesian_driver.main --experiment driving_demo

# Run medical imaging demo
python -m adaptive_bayesian_driver.main --experiment medical_imaging

# Interactive demo
python -m adaptive_bayesian_driver.main --demo
```

### Configuration

Customize experiments using `config/experiment.yaml`:

```yaml
experiment:
  name: "driving_demo"
  num_trials: 1000
  save_results: true

environment:
  volatility_params:
    hazard_rate: 0.005
    tau_drift: 30
  task_params:
    danger_probability: 0.8
    difficulty_levels: 3

model:
  inference_mode: "dual"  # "particle", "variational", or "dual"
  particle_count: 100
  learning_rate: 0.01
```

## 🧠 Core Concepts

### Surprise Detection
The system implements multi-modal surprise detection inspired by the LC-NE system:

```python
from adaptive_bayesian_driver.models.surprise import SurpriseMeter

surprise_meter = SurpriseMeter(
    metrics=['reconstruction', 'kl_divergence', 'predictive']
)
surprise_score = surprise_meter.compute_surprise(observation, prediction)
```

### Recursive Bayesian Learning
Dual-mode inference combines particle filtering with variational methods:

```python
from adaptive_bayesian_driver.models.recursive_bayesian import RecursiveBayesianLearner

learner = RecursiveBayesianLearner(
    inference_mode='dual',
    particle_count=100,
    obs_dim=784,
    latent_dim=10
)
```

### Environment Simulation
Configurable environments for different domains:

```python
from adaptive_bayesian_driver.environment.scene_renderer import SceneRenderer

renderer = SceneRenderer(
    scene_type='driving',
    difficulty_level=2,
    noise_level=0.1
)
scene = renderer.render(trial_data)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_environment.py
pytest tests/test_recursive_bayes.py

# Run with coverage
pytest --cov=adaptive_bayesian_driver tests/
```

## 📊 Applications

### Autonomous Driving
- Fork intersection navigation with adaptive decision-making
- Danger sign detection and avoidance
- Uncertainty-aware path planning

### Medical Imaging
- Longitudinal MRI analysis for anomaly detection
- Adaptive thresholding for lesion identification
- Uncertainty quantification in diagnoses

### Research Applications
- Computational neuroscience modeling
- Adaptive learning algorithms
- Non-stationary environment navigation

## 🎯 Key Components

### Models
- **Base Models**: Generative models, VAE, particle filters
- **Recursive Bayesian**: Dual-mode inference with uncertainty estimation
- **Surprise Detection**: Multi-modal surprise metrics
- **Particle Utilities**: Efficient particle filter operations

### Environment
- **Volatility Control**: HMM-based environmental changes
- **Scene Rendering**: Configurable visual environments
- **Task HMM**: Hidden state modeling for complex tasks

### Applications
- **Medical Imaging**: MRI processing and anomaly detection
- **Visualization**: Comprehensive plotting and analysis tools

## 🔧 Development

### Development Guidelines
This project follows established development practices. See these files for detailed guidance:

- **[REPO_GUIDELINES.md](REPO_GUIDELINES.md)**: Comprehensive development standards
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Quick command reference
- **[_reports/README.md](_reports/README.md)**: Report generation standards
- **[_assistant/README.md](_assistant/README.md)**: AI assistant guidelines and session management

### Project Structure
The project follows modern Python packaging standards:
- Type hints throughout
- Comprehensive testing
- CI/CD with GitHub Actions
- Configuration management
- Professional documentation

### Contributing
1. Review [REPO_GUIDELINES.md](REPO_GUIDELINES.md) for development standards
2. Fork the repository
3. Create a feature branch following naming conventions
4. Add tests for new functionality
5. Update documentation and changelog
6. Generate appropriate reports
7. Ensure all tests pass
8. Submit a pull request with attached reports

### Session Continuity
For AI-assisted development sessions, reference these resources:
- **[_assistant/README.md](_assistant/README.md)**: AI assistant guidelines overview
- **[_assistant/SESSION_GUIDELINES.md](_assistant/SESSION_GUIDELINES.md)**: Complete session procedures
- **[_assistant/QUICK_COMMANDS.md](_assistant/QUICK_COMMANDS.md)**: Command reference for assistants
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: General quick reference for project state and commands

## 📈 Performance

The system demonstrates:
- **Adaptive Learning**: Rapid adaptation to environmental changes
- **Uncertainty Quantification**: Reliable confidence estimates
- **Robust Performance**: Stable operation across different scenarios
- **Scalability**: Efficient processing of large datasets

## 🔗 References

- Locus Coeruleus-Norepinephrine system in adaptive learning
- Recursive Bayesian inference for non-stationary environments
- Surprise-based learning and attention mechanisms
- Variational inference and particle filtering methods

## 📄 License

MIT License - see LICENSE file for details.

## � Acknowledgments

This project builds upon research in computational neuroscience, adaptive learning, and uncertainty quantification. Special thanks to the biological inspiration from the LC-NE system and the broader machine learning community.

- **Research Background**: [LC-NE system modeling papers]
- **Technical Documentation**: [API reference]
- **Demo Video**: [YouTube demonstration]
- **Related Work**: [Autonomous driving perception]
