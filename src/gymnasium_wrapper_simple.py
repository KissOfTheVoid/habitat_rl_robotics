def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step using Habitat's step method."""
        # Convert to Habitat action format  
        habitat_action = self._flatten_action(action)
        
        # Use Habitat's standard step method
        observations = self._env.step(habitat_action)
        
        # Get measurements and metrics
        metrics = self._env.task.measurements.get_metrics()
        reward = metrics.get(self.habitat_config.habitat.task.reward_measure, 0.0)
        success = metrics.get(self.habitat_config.habitat.task.success_measure, 0)
        
        self._episode_step += 1
        self._episode_reward += reward
        
        # Termination
        terminated = bool(success) and self.habitat_config.habitat.task.end_on_success
        truncated = self._episode_step >= self.max_episode_steps
        
        # Process observation
        observation = self._process_observation(observations)
        
        # Info dict
        info = {
            'episode_reward': self._episode_reward,
            'episode_length': self._episode_step,
        }
        info.update(metrics)
        
        return observation, reward, terminated, truncated, info
