#!/usr/bin/env python3
"""Test SceneRenderer directly to understand the issue."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from adaptive_bayesian_driver.environment import SceneRenderer

# Test configuration
config = {
    'visual_parameters': {
        'scene_size': [64, 64],
        'safe_background_color': [50, 50, 50],
        'danger_background_color': [255, 215, 0]
    }
}

renderer = SceneRenderer(config)

# Test parameters
params = {
    'has_danger': True,
    'orientation_angle': 45.0,
    'difficulty': 0.2,
    'correct_direction': 'right'
}

scene, metadata = renderer.render_scene(params)

print("Test Results:")
print(f"Scene shape: {scene.shape}")
print(f"Scene type: {type(scene)}")
print(f"Expected shape: (1, 64, 64, 3)")
print(f"Assertion would pass: {scene.shape == (1, 64, 64, 3)}")
print(f"Metadata: {metadata}")

# Test if it's actually (1, 64, 64) without the 3
print(f"Scene shape equals (1, 64, 64): {scene.shape == (1, 64, 64)}")
print(f"Scene shape equals (1, 64, 64, 3): {scene.shape == (1, 64, 64, 3)}")
