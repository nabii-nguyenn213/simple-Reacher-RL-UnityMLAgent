import numpy as np
from envs.ReacherEnvironment import UnityReacherEnv

def greedy_action_from_obs(obs: np.ndarray) -> np.ndarray:
    # obs layout:
    # [agent_x, agent_y, agent_z,
    #  target_x, target_y, target_z,
    #  rel_x, rel_y, rel_z]
    rel_x = obs[6]
    rel_z = obs[8]

    action = np.array([rel_x, rel_z], dtype=np.float32)
    norm = np.linalg.norm(action)

    if norm > 1e-8:
        action = action / max(1.0, norm)

    return np.clip(action, -1.0, 1.0)


def main():
    env = UnityReacherEnv(
        file_name=None,
        time_scale=5.0,
        no_graphics=False,
    )
    obs = env.reset()
    print("=== TEST: success condition + reset ===")
    print("behavior_name =", env.behavior_name)
    print("obs_dim =", env.obs_dim)
    print("act_dim =", env.act_dim)
    start_agent = obs[0:3].copy()
    start_target = obs[3:6].copy()
    print("start agent pos :", start_agent)
    print("start target pos:", start_target)
    reached = False
    terminal_reward = None
    terminal_info = None
    for step in range(300):
        action = greedy_action_from_obs(obs)
        next_obs, reward, done, info = env.step(action)
        print(
            f"step={step:03d} "
            f"action={action} "
            f"reward={reward:.3f} "
            f"done={done} "
            f"interrupted={info['interrupted']}"
        )
        obs = next_obs
        if done:
            reached = reward > 0.5 and not info["interrupted"]
            terminal_reward = reward
            terminal_info = info
            break

    if not reached:
        print("\n[FAIL] Agent did not reach target successfully.")
        print("terminal_reward =", terminal_reward)
        print("terminal_info   =", terminal_info)
        env.close()
        return
    print("\n[PASS] Success condition works.")
    print("Terminal reward is positive and done=True.")
    pre_reset_obs = obs.copy()
    obs = env.reset()
    new_agent = obs[0:3]
    new_target = obs[3:6]
    print("\nAfter reset:")
    print("new agent pos :", new_agent)
    print("new target pos:", new_target)
    same_agent = np.allclose(start_agent, new_agent)
    same_target = np.allclose(start_target, new_target)
    if same_agent and same_target:
        print("[FAIL] Reset did not appear to change agent/target positions.")
    else:
        print("[PASS] Reset starts a new episode with new positions.")
    env.close()

if __name__ == "__main__":
    main()
