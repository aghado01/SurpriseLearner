"""
Test suite for environment components.
"""

import unittest
import numpy as np
import torch
from adaptive_bayesian_driver.environment import (
    VolatilityController, VolatilityRegime,
    SceneRenderer, TaskHMM, TaskState
)
from adaptive_bayesian_driver.config import load_config

class TestVolatilityController(unittest.TestCase):
    """Test VolatilityController functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'volatility_parameters': {
                'hazard_rate': 0.01,
                'tau_drift': 30,
                'bias_extremes': [0.2, 0.8]
            }
        }
        self.controller = VolatilityController(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.controller.current_bias, 0.5)
        self.assertEqual(self.controller.target_bias, 0.5)
        self.assertEqual(self.controller.volatility_regime, VolatilityRegime.STABLE)

    def test_update(self):
        """Test update functionality."""
        result = self.controller.update(0)

        self.assertIn('current_bias', result)
        self.assertIn('target_bias', result)
        self.assertIn('volatility_regime', result)
        self.assertIn('context_switched', result)

    def test_bias_bounds(self):
        """Test bias stays within bounds."""
        for _ in range(100):
            self.controller.update(0)
            self.assertGreaterEqual(self.controller.current_bias, 0.0)
            self.assertLessEqual(self.controller.current_bias, 1.0)

    def test_reset(self):
        """Test reset functionality."""
        # Change state
        for _ in range(10):
            self.controller.update(0)

        # Reset
        self.controller.reset()

        self.assertEqual(self.controller.current_bias, 0.5)
        self.assertEqual(len(self.controller.bias_history), 0)

class TestSceneRenderer(unittest.TestCase):
    """Test SceneRenderer functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'visual_parameters': {
                'scene_size': [64, 64],
                'safe_background_color': [50, 50, 50],
                'danger_background_color': [255, 215, 0]
            }
        }
        self.renderer = SceneRenderer(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.renderer.scene_size, [64, 64])
        self.assertEqual(self.renderer.safe_color, [50, 50, 50])

    def test_render_scene(self):
        """Test scene rendering."""
        params = {
            'has_danger': True,
            'orientation_angle': 45.0,
            'difficulty': 0.2,
            'correct_direction': 'right'
        }

        scene, metadata = self.renderer.render_scene(params)

        # Check output format
        self.assertIsInstance(scene, torch.Tensor)
        self.assertEqual(scene.shape, (1, 64, 64, 3))  # RGB channels
        self.assertIsInstance(metadata, dict)

        # Check metadata
        self.assertEqual(metadata['has_danger'], True)
        self.assertEqual(metadata['correct_direction'], 'right')

    def test_scene_sequence(self):
        """Test scene sequence generation."""
        scenes, metadata_list = self.renderer.create_scene_sequence(
            n_scenes=10, bias=0.7, danger_probability=0.8
        )

        self.assertEqual(scenes.shape[0], 10)  # Batch size
        self.assertEqual(len(metadata_list), 10)

    def test_difficulty_application(self):
        """Test difficulty effects."""
        params_easy = {
            'has_danger': False,
            'orientation_angle': 0.0,
            'difficulty': 0.0,
            'correct_direction': 'right'
        }

        params_hard = {
            'has_danger': False,
            'orientation_angle': 0.0,
            'difficulty': 0.8,
            'correct_direction': 'right'
        }

        scene_easy, _ = self.renderer.render_scene(params_easy)
        scene_hard, _ = self.renderer.render_scene(params_hard)

        # Hard scene should be different due to noise/occlusion
        self.assertFalse(torch.equal(scene_easy, scene_hard))

class TestTaskHMM(unittest.TestCase):
    """Test TaskHMM functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'hmm_learning_rate': 0.01,
            'hmm_adaptation': True
        }
        self.hmm = TaskHMM(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.hmm.n_states, len(TaskState))
        self.assertEqual(self.hmm.current_state, 0)
        self.assertEqual(self.hmm.transition_matrix.shape, (self.hmm.n_states, self.hmm.n_states))

    def test_state_update(self):
        """Test state update."""
        observation = {
            'has_danger': True,
            'orientation_angle': 90.0,
            'correct_direction': 'left'
        }

        result = self.hmm.update_state(observation)

        self.assertIn('current_state', result)
        self.assertIn('state_name', result)
        self.assertIn('state_probabilities', result)

    def test_observation_generation(self):
        """Test observation generation."""
        observation = self.hmm.generate_observation()

        self.assertIn('has_danger', observation)
        self.assertIn('orientation_angle', observation)
        self.assertIn('correct_direction', observation)

    def test_likelihood_computation(self):
        """Test likelihood computation."""
        observations = [
            {'has_danger': True, 'orientation_angle': 90.0, 'correct_direction': 'left'},
            {'has_danger': False, 'orientation_angle': 0.0, 'correct_direction': 'right'}
        ]

        likelihood = self.hmm.compute_likelihood(observations)
        self.assertIsInstance(likelihood, float)

    def test_reset(self):
        """Test reset functionality."""
        # Process some observations
        for i in range(5):
            obs = self.hmm.generate_observation()
            self.hmm.update_state(obs)

        # Reset
        self.hmm.reset()

        self.assertEqual(self.hmm.current_state, 0)
        self.assertEqual(len(self.hmm.state_history), 0)

if __name__ == '__main__':
    unittest.main()
