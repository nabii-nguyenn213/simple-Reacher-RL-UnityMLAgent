from __future__ import annotations
import os
import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from agents.sac_agent import SACAgent, SACConfig
from buffers.replay_buffer import ReplayBuffer
from envs.ReacherEnvironment import UnityReacherEnv
from utils.logger import TensorboardLogger

@dataclass
class TrainerConfig:
    total_timesteps: int = 100_000
    buffer_capacity: int = 200_000
    batch_size: int = 256
    learning_starts: int = 2_000
    train_every: int = 1
    gradient_steps: int = 1
    random_exploration_steps: int = 2_000
    seed: int = 42
    worker_id: int = 0
    time_scale: float = 10.0
    no_graphics: bool = False
    timeout_wait: int = 60
    save_every_steps: int = 10_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    experiment_name: str = "sac_reacher"
    # Unity Editor mode
    file_name: str | None = None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    trainer_config = TrainerConfig()
    set_seed(trainer_config.seed)
    env = UnityReacherEnv(
        file_name=trainer_config.file_name,
        seed=trainer_config.seed,
        worker_id=trainer_config.worker_id,
        time_scale=trainer_config.time_scale,
        no_graphics=trainer_config.no_graphics,
        timeout_wait=trainer_config.timeout_wait,
    )

    logger = TensorboardLogger(
        log_dir=trainer_config.log_dir,
        experiment_name=trainer_config.experiment_name,
        timestamp=True,
    )

    print("behavior_name =", env.behavior_name)
    print("obs_dim =", env.obs_dim)
    print("act_dim =", env.act_dim)
    print("tensorboard_log_dir =", logger.log_dir)

    agent = SACAgent(
        SACConfig(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            hidden_dims=(256, 256),
            activation="ReLU",
            state_independent_log_std=False,
            init_log_std=-0.5,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            auto_alpha=True,
            alpha=0.2,
            target_entropy=None,
            device="auto",
        )
    )

    replay_buffer = ReplayBuffer(
        capacity=trainer_config.buffer_capacity,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
    )

    os.makedirs(trainer_config.checkpoint_dir, exist_ok=True)

    logger.log_text("config/trainer", str(trainer_config))
    logger.log_text("config/agent", str(agent.config))

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0
    recent_episode_rewards = deque(maxlen=20)

    last_metrics: dict[str, float] = {}

    try:
        for global_step in range(1, trainer_config.total_timesteps + 1):
            logger.log_scalar("sac/train/global_step", global_step, global_step)

            if global_step <= trainer_config.random_exploration_steps:
                action = env.sample_random_action()
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, done, info = env.step(action)

            replay_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if (
                global_step >= trainer_config.learning_starts
                and global_step % trainer_config.train_every == 0
                and len(replay_buffer) >= trainer_config.batch_size
            ):
                for _ in range(trainer_config.gradient_steps):
                    last_metrics = agent.update(
                        replay_buffer=replay_buffer,
                        batch_size=trainer_config.batch_size,
                        logger=logger,
                        step=global_step,
                    )

            if done:
                recent_episode_rewards.append(episode_reward)

                avg_reward = (
                    sum(recent_episode_rewards) / len(recent_episode_rewards)
                    if recent_episode_rewards
                    else 0.0
                )

                logger.log_scalar("sac/episode/reward", episode_reward, global_step)
                logger.log_scalar("sac/episode/length", episode_length, global_step)
                logger.log_scalar("sac/train/avg_reward_20", avg_reward, global_step)
                logger.log_scalar(
                    "sac/episode/interrupted",
                    1.0 if info["interrupted"] else 0.0,
                    global_step,
                )
                logger.log_scalar("sac/train/replay_size", len(replay_buffer), global_step)

                print(
                    f"[Episode] "
                    f"global_step={global_step} "
                    f"reward={episode_reward:.3f} "
                    f"length={episode_length} "
                    f"avg_reward_20={avg_reward:.3f} "
                    f"interrupted={info['interrupted']}"
                )

                if last_metrics:
                    print(
                        f"          "
                        f"actor_loss={last_metrics.get('actor_loss', 0.0):.4f} "
                        f"critic_loss={last_metrics.get('critic_loss', 0.0):.4f} "
                        f"alpha={last_metrics.get('alpha', 0.0):.4f} "
                        f"q1_mean={last_metrics.get('q1_mean', 0.0):.4f} "
                        f"q2_mean={last_metrics.get('q2_mean', 0.0):.4f}"
                    )

                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0

            if global_step % trainer_config.save_every_steps == 0:
                ckpt_path = os.path.join(
                    trainer_config.checkpoint_dir,
                    f"sac_reacher_step_{global_step}.pt",
                )
                agent.save(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        final_path = os.path.join(
            trainer_config.checkpoint_dir,
            "sac_reacher_final.pt",
        )
        agent.save(final_path)
        print(f"Training finished. Final checkpoint saved to {final_path}")

    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
