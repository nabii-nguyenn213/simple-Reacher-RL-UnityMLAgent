import numpy as np
from envs.ReacherEnvironment import UnityReacherEnv

env = UnityReacherEnv(
    file_name=None,
    time_scale=5.0,
    no_graphics=False,
)

obs = env.reset()
print("behavior_name =", env.behavior_name)
print("obs_dim =", env.obs_dim)
print("act_dim =", env.act_dim)
print("initial obs shape =", obs.shape)

for step in range(50):
    action = np.array([1.0, 0.0], dtype=np.float32)
    print("sending action =", action)

    next_obs, reward, done, info = env.step(action)

    print(
        f"step={step:03d} "
        f"reward={reward:.3f} "
        f"done={done} "
        f"interrupted={info['interrupted']} "
        f"obs_shape={next_obs.shape}"
    )

    if done:
        print("Episode ended. Resetting...")
        obs = env.reset()
    else:
        obs = next_obs

env.close()
