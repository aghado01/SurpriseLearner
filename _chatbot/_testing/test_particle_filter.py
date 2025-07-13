#!/usr/bin/env python3
"""Test AdaptiveParticleFilter reset functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from adaptive_bayesian_driver.models import AdaptiveParticleFilter
import numpy as np

def test_particle_filter_reset():
    """Test that AdaptiveParticleFilter reset works without IndexError."""
    print("Testing AdaptiveParticleFilter reset...")

    # Create config
    config = {
        'n_particles': 50,
        'state_dim': 3,
        'observation_dim': 2,
        'process_noise_std': 0.1,
        'observation_noise_std': 0.1,
        'adaptive_noise': True,
        'adaptive_resampling': True
    }

    # Create particle filter
    pf = AdaptiveParticleFilter(config)

    print(f"Initial particles shape: {pf.particles.shape}")
    print(f"Initial weights shape: {pf.weights.shape}")

    # Run some updates to trigger potential resampling
    for i in range(5):
        observation = np.random.randn(2)
        print(f"Update {i+1}: observation = {observation}")
        try:
            result = pf.update(observation)
            print(f"  Update successful, ESS: {result['effective_sample_size']:.2f}")
        except Exception as e:
            print(f"  Update failed: {e}")
            return False

    # Test reset
    print("\nTesting reset...")
    try:
        pf.reset()
        print("Reset successful!")

        # Verify reset worked
        assert len(pf.state_history) == 0, "State history should be empty after reset"
        assert abs(np.sum(pf.weights) - 1.0) < 1e-6, "Weights should sum to 1.0"
        assert pf.particles.shape == (50, 3), "Particles shape should be preserved"

        print("All reset checks passed!")
        return True

    except Exception as e:
        print(f"Reset failed: {e}")
        return False

if __name__ == "__main__":
    success = test_particle_filter_reset()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
    sys.exit(0 if success else 1)
