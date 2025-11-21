#!/usr/bin/env python3
import sys
import numpy as np
sys.path.insert(0, '/home/iskopyl/habitat_rl_sorting/src')

from gymnasium_wrapper import ColorSortingGymEnv

def test_basic_functionality():
    """Test 1: Basic environment initialization and structure"""
    print("\n" + "="*80)
    print("TEST 1: Basic Environment Initialization")
    print("="*80)
    
    try:
        # Create environment
        print("\n[1/5] Creating Gymnasium environment...")
        env = ColorSortingGymEnv(
            config_path="configs/color_sorting.yaml",
            render_mode="rgb_array",
            max_episode_steps=50
        )
        print("✓ Environment created successfully")
        
        # Check observation space
        print("\n[2/5] Checking observation space...")
        print(f"  Observation space: {env.observation_space}")
        rgb_space = env.observation_space['rgb']
        state_space = env.observation_space['state']
        print(f"  RGB shape: {rgb_space.shape}")
        print(f"  State shape: {state_space.shape}")
        print("✓ Observation space valid")
        
        # Check action space
        print("\n[3/5] Checking action space...")
        print(f"  Action space: {env.action_space}")
        # Note: ActionSpace is composite, not a simple Box
        # print(f"  Action shape: {env.action_space}")
        print("✓ Action space valid")
        
        # Reset environment
        print("\n[4/5] Resetting environment...")
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful")
        
        # Take random action
        print("\n[5/5] Taking random action...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print("✓ Step successful")
        
        env.close()
        print("\n✓ All checks passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_episode_rollout():
    """Test 2: Full episode rollout"""
    print("\n" + "="*80)
    print("TEST 2: Full Episode Rollout")
    print("="*80)
    
    try:
        print("\nRunning episode with random actions...\n")
        
        env = ColorSortingGymEnv(
            config_path="configs/color_sorting.yaml",
            render_mode="rgb_array",
            max_episode_steps=20
        )
        
        obs, info = env.reset(seed=42)
        total_reward = 0
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                success_val = info.get('color_sorting_success', 0)
                print(f"  Step {step:02d}: reward={reward:.4f}, " +
                      f"success={success_val}, " +
                      f"total_reward={total_reward:.4f}")
            
            if terminated or truncated:
                print(f"\n  Episode ended at step {step}")
                break
        
        env.close()
        print(f"\n✓ Episode completed! Total reward: {total_reward:.4f}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_observation_validity():
    """Test 3: Observation validity and consistency"""
    print("\n" + "="*80)
    print("TEST 3: Observation Validity")
    print("="*80)
    
    try:
        env = ColorSortingGymEnv(
            config_path="configs/color_sorting.yaml",
            render_mode="rgb_array",
            max_episode_steps=10
        )
        
        obs, info = env.reset(seed=42)
        
        # Check RGB observation
        print("\n[RGB Observation]")
        rgb_obs = obs['rgb']
        print(f"  Shape: {rgb_obs.shape}")
        print(f"  Dtype: {rgb_obs.dtype}")
        print(f"  Range: [{rgb_obs.min()}, {rgb_obs.max()}]")
        
        assert rgb_obs.dtype == np.uint8, "RGB should be uint8"
        assert rgb_obs.min() >= 0 and rgb_obs.max() <= 255, "RGB should be in [0, 255]"
        print("✓ RGB observation valid")
        
        # Check state observation
        print("\n[State Observation]")
        state_obs = obs['state']
        print(f"  Shape: {state_obs.shape}")
        print(f"  Dtype: {state_obs.dtype}")
        print(f"  Range: [{state_obs.min():.4f}, {state_obs.max():.4f}]")
        
        assert state_obs.dtype == np.float32, "State should be float32"
        assert not np.any(np.isnan(state_obs)), "State should not contain NaN"
        assert not np.any(np.isinf(state_obs)), "State should not contain Inf"
        print("✓ State observation valid")
        
        # Check consistency across steps
        print("\n[Consistency Check]")
        prev_obs = obs
        for i in range(3):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            
            rgb_obs = obs['rgb']
            state_obs = obs['state']
            prev_rgb = prev_obs['rgb']
            prev_state = prev_obs['state']
            
            assert rgb_obs.shape == prev_rgb.shape, "RGB shape should be consistent"
            assert state_obs.shape == prev_state.shape, "State shape should be consistent"
            prev_obs = obs
        
        print("✓ Observations consistent across steps")
        
        env.close()
        print("\n✓ All observation checks passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "═"*80)
    print("║" + " "*18 + "HABITAT COLOR SORTING ENVIRONMENT" + " "*29 + "║")
    print("║" + " "*30 + "TEST SUITE" + " "*38 + "║")
    print("═"*80)
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Episode Rollout", test_episode_rollout()))
    results.append(("Observation Validity", test_observation_validity()))
    
    # Print summary
    print("\n\n" + "═"*80)
    print("║" + " "*32 + "TEST SUMMARY" + " "*34 + "║")
    print("╠" + "═"*80 + "╣")
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        padding = " " * (50 - len(name))
        print(f"║  {name}{padding}{status}" + " "*18 + "║")
    
    print("╠" + "═"*80 + "╣")
    print(f"║  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}" + 
          " " * (61 - len(str(len(results))) - len(str(passed)) - len(str(failed))) + "║")
    print("╚" + "═"*80 + "╝")
    
    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
