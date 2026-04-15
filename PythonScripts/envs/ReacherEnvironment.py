import numpy as np 

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityReacherEnv: 
    def __init__(self, file_name=None, seed=42, worker_id=0, time_scale=5.0, no_graphics=False, timeout_wait=60): 
        self.engine_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
                file_name=file_name,  # If `None` -> connect to Unity Editor
                seed=seed, 
                worker_id=worker_id, 
                no_graphics=no_graphics, 
                timeout_wait=timeout_wait, 
                side_channels=[self.engine_channel]
                )
        self.engine_channel.set_configuration_parameters(time_scale=time_scale)
        self.env.reset()
        behavior_names = list(self.env.behavior_specs.keys())
        if len(behavior_names) == 0: 
            raise RuntimeError("No behaviors found. Check Behavior Parameters in Unity")
        self.behavior_name=behavior_names[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        # Observation and action info 
        self.obs_specs = self.spec.observation_specs
        self.action_spec = self.spec.action_spec

        if len(self.obs_specs) == 0 :
            raise RuntimeError("No observations found for this behavior")
        if self.action_spec.continuous_size <=0 : 
            raise RuntimeError("Expected a continuous action space, but continuous size <= 0")
        # Obs_dim = 9
        # Act_dim = 2 (Continuous) 
        self.obs_dim = int(np.prod(self.obs_specs[0].shape)) 
        self.act_dim = int(self.action_spec.continuous_size) 

        self.agent_id = None 

    def reset(self): 
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(decision_steps.agent_id) == 0: 
            raise RuntimeError("No active agent requested a decision after reset()."
                               "Check Decision Requester, Agent setup, and Behavior Parameters"
                               )
        self.agent_id = int(decision_steps.agent_id[0])
        obs = decision_steps[self.agent_id].obs[0]
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def step(self, action):
        if self.agent_id is None: 
            raise RuntimeError("Call reset() before step()")
        action = np.asarray(action, dtype=np.float32).reshape(1, -1) 
        if action.shape[1] != self.act_dim: 
            raise ValueError(f"Action shape mismatch. Expected ({1}, {self.act_dim})"
                             f"got {action.shape}"
                             )
        self.env.set_actions(self.behavior_name, ActionTuple(continuous=action))
        self.env.step() 
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        info = {} 
        if self.agent_id in terminal_steps: 
            ts = terminal_steps[self.agent_id] 
            next_obs = np.asarray(ts.obs[0], dtype=np.float32).reshape(-1)
            reward = float(ts.reward)
            done = True 
            info["interrupted"] = bool(ts.interrupted)
        else: 
            if self.agent_id not in decision_steps: 
                raise RuntimeError("Tracked agent_id not found in DecisionSteps or TerminalSteps")
            ds = decision_steps[self.agent_id]
            next_obs = np.asarray(ds.obs[0], dtype=np.float32).reshape(-1)
            reward = float(ds.reward)
            done=False 
            info["interrupted"] = False 
        return next_obs, reward, done, info 

    def sample_random_action(self): 
        return np.random.uniform(-1.0, 1.0, size=(self.act_dim, )).astype(np.float32)

    def close(self): 
        self.env.close()


