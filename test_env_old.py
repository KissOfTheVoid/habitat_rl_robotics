#!/usr/bin/env python3
"""
Test script for Color Sorting Environment
Tests the full pipeline: Task, Sensors, Measurements, Gymnasium Wrapper
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))
sys.path.insert(0, os.path.expanduser("~/habitat_rl_sorting/src"))

# Import custom modules
from color_sorting_task import ColorSortingTaskV1
from color_sorting_sensors import (
    ColorZoneSensor, 
    ObjectColorSensor,
    ColorSortingSuccess,
    ColorSortingReward,
    NumObjectsCorrectlyPlaced
)
from gymnasium_wrapper import ColorSortingGymEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 80)
    print("TEST 1: Basic Environment Initialization")
    print("=" * 80)
    
    config_path = os.path.expanduser("~/habitat_rl_sorting/configs/color_sorting.yaml")
    
    try:
        # Create environment
        print("\n[1/5] Creating Gymnasium environment...")
        env = ColorSortingGymEnv(
            config_path=config_path,
            render_mode="rgb_array",
            max_episode_steps=50
        )
        print("✓ Environment created successfully")
        
        # Check observation space
        print("\n[2/5] Checking observation space...")
        print(f"  Observation space: {env.observation_space}")
        rgb_space = env.observation_space["rgb"]
        print(f"  RGB shape: {rgb_space.shape}")
        state_space = env.observation_space["state"]
        print(f"  State shape: {state_space.shape}")
        print("✓ Observation space valid")
        
        # Check action space
        print("\n[3/5] Checking action space...")
        print(f"  Action space: {env.action_space}")
        print(f"  Action shape: {env.action_space.shape}")
        print("✓ Action space valid")
        
        # Reset environment
        print("\n[4/5] Resetting environment...")
        obs, info = env.reset(seed=42)
        print(f"  Observation keys: {obs.keys()}")
        print(f"  RGB min/max: {obs["rgb"].min()}/{obs["rgb"].max()}")
        print(f"  State shape: {obs["state"].shape}")
        print(f"  Info: {info}")
        print("✓ Environment reset successful")
        
        # Take a few steps
        print("\n[5/5] Testing environment steps...")
        total_reward = 0.0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"  Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                break
        
        print(f"  Total reward: {total_reward:.4f}")
        print("✓ Environment steps successful")
        
        # Close environment
        env.close()
        print("\n✓ Environment closed successfully")
        
        print("\n" + "=" * 80)
        print("TEST 1: PASSED ✓")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_rollout():
    """Test full episode rollout."""
    print("\n" + "=" * 80)
    print("TEST 2: Full Episode Rollout")
    print("=" * 80)
    
    config_path = os.path.expanduser("~/habitat_rl_sorting/configs/color_sorting.yaml")
    
    try:
        env = ColorSortingGymEnv(
            config_path=config_path,
            render_mode="rgb_array",
            max_episode_steps=100
        )
        
        obs, info = env.reset(seed=123)
        
        print(f"\nRunning episode with random actions...")
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.3f}, "
                      f"success={info.get("color_sorting_success", 0)}, "
                      f"num_placed={info.get(num_objects_correctly_placed, 0)}")
            
            if terminated or truncated:
                print(f"\nEpisode finished:")
                print(f"  Reason: {Success if terminated else Truncated}")
                break
        
        print(f"\nEpisode statistics:")
        print(f"  Length: {episode_length} steps")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Average reward: {episode_reward/episode_length:.3f}")
        print(f"  Success: {info.get("color_sorting_success", 0)}")
        
        env.close()
        
        print("\n" + "=" * 80)
        print("TEST 2: PASSED ✓")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_validity():
    """Test that observations are valid and consistent."""
    print("\n" + "=" * 80)
    print("TEST 3: Observation Validity")
    print("=" * 80)
    
    config_path = os.path.expanduser("~/habitat_rl_sorting/configs/color_sorting.yaml")
    
    try:
        env = ColorSortingGymEnv(
            config_path=config_path,
            render_mode="rgb_array",
            max_episode_steps=10
        )
        
        obs, info = env.reset(seed=456)
        
        # Check RGB observation
        print("\n[RGB Observation]")
        print(f"  Shape: {obs["rgb"].shape}")
        print(f"  Dtype: {obs["rgb"].dtype}")
        print(f"  Range: [{obs["rgb"].min()}, {obs["rgb"].max()}]")
        assert obs["rgb"].dtype == np.uint8, "RGB should be uint8"
        assert obs["rgb"].min() >= 0 and obs["rgb"].max() <= 255, "RGB should be in [0, 255]"
        print("  ✓ RGB observation valid")
        
        # Check state observation
        print("\n[State Observation]")
        print(f"  Shape: {obs["state"].shape}")
        print(f"  Dtype: {obs["state"].dtype}")
        print(f"  Range: [{obs["state"].min():.3f}, {obs["state"].max():.3f}]")
        assert obs["state"].dtype == np.float32, "State should be float32"
        assert not np.any(np.isnan(obs["state"])), "State should not contain NaN"
        assert not np.any(np.isinf(obs["state"])), "State should not contain Inf"
        print("  ✓ State observation valid")
        
        # Check consistency across steps
        print("\n[Consistency Check]")
        prev_obs = obs
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs["rgb"].shape == prev_obs["rgb"].shape, "RGB shape should be consistent"
            assert obs["state"].shape == prev_obs["state"].shape, "State shape should be consistent"
            
            prev_obs = obs
        
        print("  ✓ Observations consistent across 5 steps")
        
        env.close()
        
        print("\n" + "=" * 80)
        print("TEST 3: PASSED ✓")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HABITAT COLOR SORTING ENVIRONMENT" + " " * 25 + "║")
    print("║" + " " * 30 + "TEST SUITE" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Episode Rollout", test_episode_rollout()))
    results.append(("Observation Validity", test_observation_validity()))
    
    # Print summary
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "TEST SUMMARY" + " " * 34 + "║")
    print("╠" + "=" * 78 + "╣")
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"║  {test_name:50s} {status:25s} ║")
    
    print("╠" + "=" * 78 + "╣")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"║  Total: {total}  |  Passed: {passed}  |  Failed: {total - passed}" + " " * 42 + "║")
    print("╚" + "=" * 78 + "╝")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready for RL training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
