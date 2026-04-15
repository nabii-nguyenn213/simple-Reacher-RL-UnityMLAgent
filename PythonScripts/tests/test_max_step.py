import numpy as np
from envs.ReacherEnvironment import UnityReacherEnv

def main():
    env = UnityReacherEnv(
        file_name=None,
        time_scale=5.0,
        no_graphics=False,
    )
    obs = env.reset()
    print("=== TEST: max step interruption ===")
    print("Make sure Max Step is set in Unity Behavior Parameters.")
    action = np.array([0.0, 0.0], dtype=np.float32)
    for step in range(400):
        next_obs, reward, done, info = env.step(action)
        print(
            f"step={step:03d} "
            f"reward={reward:.3f} "
            f"done={done} "
            f"interrupted={info['interrupted']}"
        )
        obs = next_obs
        if done:
            if info["interrupted"]:
                print("\n[PASS] Episode ended by Max Step.")
            else:
                print("\n[FAIL] Episode ended, but not by interruption.")
            env.close()
            return
    print("\n[FAIL] Episode did not end within 400 steps.")
    print("Check whether Max Step is set in Unity.")
    env.close()

if __name__ == "__main__":
    main()
