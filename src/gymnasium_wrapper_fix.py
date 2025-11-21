# Patch for gymnasium_wrapper.py lines 68-77

        # Load Habitat config
        self.habitat_config = get_config(config_path, overrides=override_options or [])
        
        # Use OmegaConf read_write context to modify read-only config
        with OmegaConf.read_write(self.habitat_config):
            with OmegaConf.set_struct(self.habitat_config, False):
                self.habitat_config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = False
                self.habitat_config.habitat.environment.max_episode_steps = max_episode_steps
        
        # Create Habitat environment
        self._env = HabitatEnv(config=self.habitat_config)
