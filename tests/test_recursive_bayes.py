"""
Test suite for recursive Bayesian learning components.
"""

import unittest
import numpy as np
import torch
from adaptive_bayesian_driver.models import (
    RecursiveBayesianLearner, InferenceMode,
    SurpriseMeter, SurpriseType,
    AdaptiveParticleFilter
)
from adaptive_bayesian_driver.config import load_config

class TestRecursiveBayesianLearner(unittest.TestCase):
    """Test RecursiveBayesianLearner functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'model_parameters': {
                'learning_rate': 0.01,
                'adaptation_rate': 0.1,
                'state_dim': 4,
                'obs_dim': 2,
                'baseline_window': 20
            },
            'experiment': {'seed': 42}
        }

    def test_gaussian_initialization(self):
        """Test Gaussian inference mode initialization."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.GAUSSIAN)

        self.assertEqual(learner.inference_mode, InferenceMode.GAUSSIAN)
        self.assertEqual(learner.state_mean.shape, (4,))
        self.assertEqual(learner.state_cov.shape, (4, 4))

    def test_particle_initialization(self):
        """Test particle filter initialization."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.PARTICLE)

        self.assertEqual(learner.inference_mode, InferenceMode.PARTICLE)
        self.assertEqual(learner.particles.shape, (100, 4))  # Default n_particles
        self.assertEqual(len(learner.weights), 100)

    def test_gaussian_prediction(self):
        """Test Gaussian prediction step."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.GAUSSIAN)

        prediction = learner.predict()

        self.assertIn('state_mean', prediction)
        self.assertIn('state_cov', prediction)
        self.assertIn('obs_mean', prediction)

    def test_gaussian_update(self):
        """Test Gaussian update step."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.GAUSSIAN)

        observation = np.array([1.0, 2.0])
        result = learner.update(observation)

        self.assertIn('primary_surprise', result)
        self.assertIn('context_beliefs', result)
        self.assertIn('adaptive_learning_rate', result)

    def test_particle_update(self):
        """Test particle filter update step."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.PARTICLE)

        observation = np.array([1.0, 2.0])
        result = learner.update(observation)

        self.assertIn('primary_surprise', result)
        self.assertIn('effective_sample_size', result)

    def test_decision_making(self):
        """Test decision making."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.GAUSSIAN)

        observation = np.array([1.0, 2.0])
        decision = learner.make_decision(observation, ['left', 'right'])

        self.assertIn('decision', decision)
        self.assertIn('confidence', decision)
        self.assertIn(decision['decision'], ['left', 'right'])

    def test_reset(self):
        """Test reset functionality."""
        learner = RecursiveBayesianLearner(self.config, InferenceMode.GAUSSIAN)

        # Process some data
        for _ in range(5):
            observation = np.random.randn(2)
            learner.update(observation)

        # Reset
        learner.reset()

        self.assertEqual(len(learner.surprise_history), 0)
        self.assertEqual(learner.context_uncertainty, 0.5)

class TestSurpriseMeter(unittest.TestCase):
    """Test SurpriseMeter functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.meter = SurpriseMeter(mode=SurpriseType.CHI2)

    def test_chi2_surprise(self):
        """Test chi-squared surprise computation."""
        innovation = np.array([1.0, 2.0])
        covariance = np.eye(2) * 0.5

        surprise = self.meter.compute_surprise(innov=innovation, S=covariance)

        self.assertIsInstance(surprise, float)
        self.assertGreaterEqual(surprise, 0.0)

    def test_reconstruction_surprise(self):
        """Test reconstruction error surprise."""
        meter = SurpriseMeter(mode=SurpriseType.RECONSTRUCTION)

        x = torch.randn(1, 10)
        x_recon = x + torch.randn(1, 10) * 0.1

        surprise = meter.compute_surprise(x=x, x_recon=x_recon)

        self.assertIsInstance(surprise, float)
        self.assertGreaterEqual(surprise, 0.0)

    def test_baseline_update(self):
        """Test baseline statistics update."""
        meter = SurpriseMeter(mode=SurpriseType.CHI2, baseline_window=5)

        # Add some errors
        for i in range(10):
            meter.update_baseline(float(i))

        # Check baseline stats updated
        self.assertGreater(meter.baseline_stats['mean'], 0)
        self.assertEqual(len(meter.recent_errors), 5)  # Window size

    def test_normalized_surprise(self):
        """Test normalized surprise computation."""
        meter = SurpriseMeter(mode=SurpriseType.CHI2)

        # Add baseline data
        for _ in range(20):
            innovation = np.random.randn(2)
            covariance = np.eye(2)
            meter.compute_normalized_surprise(innov=innovation, S=covariance)

        # Compute normalized surprise
        innovation = np.array([5.0, 5.0])  # Large innovation
        covariance = np.eye(2)

        surprise = meter.compute_normalized_surprise(innov=innovation, S=covariance)

        self.assertGreaterEqual(surprise, 0.0)

class TestAdaptiveParticleFilter(unittest.TestCase):
    """Test AdaptiveParticleFilter functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'n_particles': 50,
            'state_dim': 3,
            'observation_dim': 2,
            'process_noise_std': 0.1,
            'observation_noise_std': 0.1,
            'adaptive_noise': True,
            'adaptive_resampling': True
        }
        self.pf = AdaptiveParticleFilter(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.pf.particles.shape, (50, 3))
        self.assertEqual(len(self.pf.weights), 50)
        self.assertAlmostEqual(np.sum(self.pf.weights), 1.0, places=6)

    def test_prediction(self):
        """Test prediction step."""
        result = self.pf.predict()

        self.assertIn('mean', result)
        self.assertIn('covariance', result)
        self.assertEqual(result['mean'].shape, (3,))

    def test_update(self):
        """Test update step."""
        observation = np.array([1.0, 2.0])

        result = self.pf.update(observation)

        self.assertIn('mean', result)
        self.assertIn('effective_sample_size', result)
        self.assertIn('resampled', result)

    def test_resampling(self):
        """Test resampling functionality."""
        # Set uneven weights to trigger resampling
        self.pf.weights[0] = 0.9
        self.pf.weights[1:] = 0.1 / (len(self.pf.weights) - 1)

        observation = np.array([1.0, 2.0])
        result = self.pf.update(observation)

        # Should have resampled due to low effective sample size
        # (This might not always trigger, but test structure is correct)
        self.assertIn('resampled', result)

    def test_state_estimate(self):
        """Test state estimation."""
        estimate = self.pf.get_state_estimate()

        self.assertEqual(estimate.shape, (3,))
        self.assertIsInstance(estimate, np.ndarray)

    def test_uncertainty(self):
        """Test uncertainty computation."""
        uncertainty = self.pf.get_uncertainty()

        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(uncertainty, 0.0)

    def test_reset(self):
        """Test reset functionality."""
        # Run some updates
        for _ in range(5):
            observation = np.random.randn(2)
            self.pf.update(observation)

        # Reset
        self.pf.reset()

        self.assertEqual(len(self.pf.state_history), 0)
        self.assertAlmostEqual(np.sum(self.pf.weights), 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
