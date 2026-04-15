import numpy as np
from envs.ReacherEnvironment import UnityReacherEnv

def main():
    env = UnityReacherEnv(
        file_name=None,
        time_scale=5.0,
        no_graphics=False,
    )
    obs = env.reset()
    print("=== TEST: random actions ===")
    print("behavior_name =", env.behavior_name)
    print("obs_dim =", env.obs_dim)
    print("act_dim =", env.act_dim)
    prev_agent_pos = obs[0:3].copy()
    moved_count = 0
    for step in range(100):
        action = env.sample_random_action()
        next_obs, reward, done, info = env.step(action)
        agent_pos = next_obs[0:3]
        delta = np.linalg.norm(agent_pos - prev_agent_pos)
        if delta > 1e-5:
            moved_count += 1
        print(
            f"step={step:03d} "
            f"action={action} "
            f"delta={delta:.5f} "
            f"reward={reward:.3f} "
            f"done={done} "
            f"interrupted={info['interrupted']}"
        )
        prev_agent_pos = agent_pos
        obs = next_obs
        if done:
            print("Episode ended. Resetting...")
            obs = env.reset()
            prev_agent_pos = obs[0:3].copy()
    if moved_count > 0:
        print(f"\n[PASS] Agent moved on {moved_count} steps.")
    else:
        print("\n[FAIL] Agent never appeared to move.")
    env.close()

if __name__ == "__main__":
    main()
