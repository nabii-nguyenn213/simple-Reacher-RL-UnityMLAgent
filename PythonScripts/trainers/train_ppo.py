from __future__ import annotations
import os
from collections import deque
from dataclasses import dataclass
from agents.ppo_agent import PPOAgent, PPOConfig
from buffers.rollout_buffer import RolloutBuffer
from envs.ReacherEnvironment import UnityReacherEnv
from utils.logger import TensorboardLogger

@dataclass
class TrainerConfig:
    total_timesteps: int = 100_000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42
    worker_id: int = 0
    time_scale: float = 10.0
    no_graphics: bool = False
    timeout_wait: int = 60
    save_every_updates: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    experiment_name: str = "ppo_reacher"
    # Unity Editor mode
    file_name: str | None = None

def main():
    trainer_config = TrainerConfig()

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

    agent = PPOAgent(
        PPOConfig(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            hidden_dims=(128, 128),
            activation="ReLU",
            state_independent_log_std=True,
            init_log_std=-0.5,
            lr=3e-4,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            update_epochs=10,
            mini_batch_size=64,
            normalize_advantages=True,
            device="auto",
        )
    )

    buffer = RolloutBuffer(
        buffer_size=trainer_config.rollout_steps,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        gamma=trainer_config.gamma,
        gae_lambda=trainer_config.gae_lambda,
    )

    os.makedirs(trainer_config.checkpoint_dir, exist_ok=True)

    logger.log_text("config/trainer", str(trainer_config))
    logger.log_text("config/agent", str(agent.config))

    obs = env.reset()
    global_step = 0
    update_idx = 0

    episode_reward = 0.0
    episode_length = 0
    recent_episode_rewards = deque(maxlen=20)

    try:
        while global_step < trainer_config.total_timesteps:
            buffer.reset()
            last_done = False

            for _ in range(trainer_config.rollout_steps):
                env_action, raw_action, log_prob, value = agent.select_action(obs)

                next_obs, reward, done, info = env.step(env_action)

                buffer.add(
                    obs=obs,
                    raw_action=raw_action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                obs = next_obs
                global_step += 1
                episode_reward += reward
                episode_length += 1
                last_done = done

                if done:
                    recent_episode_rewards.append(episode_reward)

                    logger.log_scalar("ppo/episode/reward", episode_reward, global_step)
                    logger.log_scalar("ppo/episode/length", episode_length, global_step)
                    logger.log_scalar(
                        "ppo/episode/interrupted",
                        1.0 if info["interrupted"] else 0.0,
                        global_step,
                    )

                    print(
                        f"[Episode] "
                        f"global_step={global_step} "
                        f"reward={episode_reward:.3f} "
                        f"length={episode_length} "
                        f"interrupted={info['interrupted']}"
                    )

                    obs = env.reset()
                    episode_reward = 0.0
                    episode_length = 0

                if global_step >= trainer_config.total_timesteps:
                    break

            if last_done:
                last_value = 0.0
            else:
                _, last_value = agent.select_action_deterministic(obs)

            buffer.compute_returns_and_advantages(
                last_value=last_value,
                last_done=last_done,
            )

            metrics = agent.update(
                buffer,
                logger=logger,
                step=global_step,
            )
            update_idx += 1

            avg_reward = (
                sum(recent_episode_rewards) / len(recent_episode_rewards)
                if recent_episode_rewards
                else 0.0
            )

            logger.log_scalar("ppo/train/global_step", global_step, global_step)
            logger.log_scalar("ppo/train/update_idx", update_idx, global_step)
            logger.log_scalar("ppo/train/avg_reward_20", avg_reward, global_step)

            print(
                f"[Update {update_idx:04d}] "
                f"global_step={global_step} "
                f"avg_reward_20={avg_reward:.3f} "
                f"loss={metrics.get('loss', 0.0):.4f} "
                f"policy_loss={metrics.get('policy_loss', 0.0):.4f} "
                f"value_loss={metrics.get('value_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f} "
                f"approx_kl={metrics.get('approx_kl', 0.0):.4f} "
                f"clip_fraction={metrics.get('clip_fraction', 0.0):.4f}"
            )

            if update_idx % trainer_config.save_every_updates == 0:
                ckpt_path = os.path.join(
                    trainer_config.checkpoint_dir,
                    f"ppo_reacher_update_{update_idx}.pt",
                )
                agent.save(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        final_path = os.path.join(
            trainer_config.checkpoint_dir,
            "ppo_reacher_final.pt",
        )
        agent.save(final_path)
        print(f"Training finished. Final checkpoint saved to {final_path}")

    finally:
        logger.close()
        env.close()

if __name__ == "__main__":
    main()
