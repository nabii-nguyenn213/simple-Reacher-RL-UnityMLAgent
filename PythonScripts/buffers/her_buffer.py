from __future__ import annotations
import numpy as np
import torch

class HERReplayBuffer:
    """
    Replay buffer with Hindsight Experience Replay (HER).

    Expected flat observation layout for your Reacher env:
        obs[0:3] = achieved_goal   = agent position
        obs[3:6] = desired_goal    = target position
        obs[6:9] = relative goal   = desired_goal - achieved_goal
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        her_k: int = 4,
        goal_selection_strategy: str = "future",
        success_threshold: float = 1.0,
        step_penalty: float = -0.001,
        success_reward: float = 1.0,
        fall_penalty: float = -1.0,
        fall_y_threshold: float = -1.0,
        achieved_goal_slice: slice = slice(0, 3),
        desired_goal_slice: slice = slice(3, 6),
        relative_goal_slice: slice = slice(6, 9),
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.her_k = her_k
        self.goal_selection_strategy = goal_selection_strategy.lower()
        self.success_threshold = success_threshold

        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.fall_penalty = fall_penalty
        self.fall_y_threshold = fall_y_threshold

        self.achieved_goal_slice = achieved_goal_slice
        self.desired_goal_slice = desired_goal_slice
        self.relative_goal_slice = relative_goal_slice

        self._valid_goal_strategies = {
            "future",
            "final",
            "episode",
            "random",
            "current",
        }

        if self.goal_selection_strategy not in self._valid_goal_strategies:
            raise ValueError(
                f"Unsupported goal_selection_strategy: {self.goal_selection_strategy}. "
                f"Supported: {sorted(self._valid_goal_strategies)}"
            )

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        # HER logging stats
        self.total_original_added = 0
        self.total_her_added = 0
        self.total_episodes_added = 0

    def __len__(self) -> int:
        return self.size

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_achieved_goal(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.achieved_goal_slice].copy()

    def _set_desired_goal(self, obs: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        new_obs = obs.copy()
        achieved_goal = new_obs[self.achieved_goal_slice].copy()

        new_obs[self.desired_goal_slice] = desired_goal
        new_obs[self.relative_goal_slice] = desired_goal - achieved_goal

        return new_obs

    def _is_fall(self, obs: np.ndarray) -> bool:
        achieved_goal = self._get_achieved_goal(obs)
        y = float(achieved_goal[1])
        return y < self.fall_y_threshold

    def _is_success(self, next_obs: np.ndarray, desired_goal: np.ndarray) -> bool:
        achieved_goal = self._get_achieved_goal(next_obs)
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return dist <= self.success_threshold

    def _compute_reward_and_done(
        self,
        next_obs: np.ndarray,
        desired_goal: np.ndarray,
        original_done: bool,
    ) -> tuple[float, bool]:
        fell = self._is_fall(next_obs)
        success = self._is_success(next_obs, desired_goal)

        if success:
            reward = self.success_reward
        elif fell:
            reward = self.fall_penalty
        else:
            reward = self.step_penalty

        done = bool(success or fell or original_done)
        return reward, done

    def _sample_goal_indices(self, t: int, episode_length: int) -> list[int]:
        if self.her_k <= 0:
            return []

        strategy = self.goal_selection_strategy

        if strategy == "future":
            if t >= episode_length - 1:
                return []
            return np.random.randint(t + 1, episode_length, size=self.her_k).tolist()

        if strategy == "final":
            return [episode_length - 1 for _ in range(self.her_k)]

        if strategy in {"episode", "random"}:
            return np.random.randint(0, episode_length, size=self.her_k).tolist()

        if strategy == "current":
            return [t for _ in range(self.her_k)]

        raise ValueError(
            f"Unsupported goal_selection_strategy: {strategy}. "
            f"Supported: {sorted(self._valid_goal_strategies)}"
        )

    def add_episode(self, episode_transitions: list[dict]) -> dict[str, float]:
        """
        Each transition dict must contain:
            obs
            action
            reward
            next_obs
            done

        Returns episode-level and cumulative HER statistics.
        """
        if len(episode_transitions) == 0:
            return {
                "episode_original_added": 0.0,
                "episode_her_added": 0.0,
                "episode_total_added": 0.0,
                "episode_original_ratio": 0.0,
                "episode_her_ratio": 0.0,
                "cumulative_original_added": float(self.total_original_added),
                "cumulative_her_added": float(self.total_her_added),
                "cumulative_total_added": float(self.total_original_added + self.total_her_added),
                "cumulative_original_ratio": 0.0,
                "cumulative_her_ratio": 0.0,
                "episodes_added": float(self.total_episodes_added),
            }

        episode_length = len(episode_transitions)
        episode_original_added = 0
        episode_her_added = 0

        for t, transition in enumerate(episode_transitions):
            obs = transition["obs"]
            action = transition["action"]
            reward = float(transition["reward"])
            next_obs = transition["next_obs"]
            done = bool(transition["done"])

            # Original transition
            self._store_transition(obs, action, reward, next_obs, done)
            episode_original_added += 1

            # HER transitions
            goal_indices = self._sample_goal_indices(t, episode_length)

            for goal_idx in goal_indices:
                selected_transition = episode_transitions[goal_idx]
                new_goal = self._get_achieved_goal(selected_transition["next_obs"])

                her_obs = self._set_desired_goal(obs, new_goal)
                her_next_obs = self._set_desired_goal(next_obs, new_goal)

                her_reward, her_done = self._compute_reward_and_done(
                    next_obs=her_next_obs,
                    desired_goal=new_goal,
                    original_done=done,
                )

                self._store_transition(
                    her_obs,
                    action,
                    her_reward,
                    her_next_obs,
                    her_done,
                )
                episode_her_added += 1

        self.total_original_added += episode_original_added
        self.total_her_added += episode_her_added
        self.total_episodes_added += 1

        episode_total_added = episode_original_added + episode_her_added
        cumulative_total_added = self.total_original_added + self.total_her_added

        episode_original_ratio = (
            episode_original_added / episode_total_added if episode_total_added > 0 else 0.0
        )
        episode_her_ratio = (
            episode_her_added / episode_total_added if episode_total_added > 0 else 0.0
        )

        cumulative_original_ratio = (
            self.total_original_added / cumulative_total_added if cumulative_total_added > 0 else 0.0
        )
        cumulative_her_ratio = (
            self.total_her_added / cumulative_total_added if cumulative_total_added > 0 else 0.0
        )

        return {
            "episode_original_added": float(episode_original_added),
            "episode_her_added": float(episode_her_added),
            "episode_total_added": float(episode_total_added),
            "episode_original_ratio": float(episode_original_ratio),
            "episode_her_ratio": float(episode_her_ratio),
            "cumulative_original_added": float(self.total_original_added),
            "cumulative_her_added": float(self.total_her_added),
            "cumulative_total_added": float(cumulative_total_added),
            "cumulative_original_ratio": float(cumulative_original_ratio),
            "cumulative_her_ratio": float(cumulative_her_ratio),
            "episodes_added": float(self.total_episodes_added),
        }

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise RuntimeError(
                f"Not enough samples in HERReplayBuffer. size={self.size}, batch_size={batch_size}"
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=device,
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            "dones": torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ),
        }
