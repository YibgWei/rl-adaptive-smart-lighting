import os
import json
from statistics import mean

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from env.lighting_env import SmartLightingEnv

# Configuration for multi-seed experiment and training duration
RESULTS_DIRECTORY = "results_train"
SEEDS = [1, 2, 3]
TOTAL_TIMESTEPS = 600000

def create_monitored_environment():
    """Wraps the lighting environment in a Monitor to track episode rewards and lengths."""
    return Monitor(SmartLightingEnv())

def run_quick_test(model):
    """Evaluates the trained agent on a single test episode to gather performance data."""
    testing_environment = SmartLightingEnv()
    observation, information = testing_environment.reset(seed=0)

    cumulative_reward = 0.0
    terminated = False
    truncated = False

    # Metrics storage for post-run analysis
    comfort_scores = []
    energy_efficiency_scores = []
    colour_temperature_scores = []
    overall_scores = []

    while not (terminated or truncated):
        # Predict the best action (deterministic) based on the current observation
        selected_action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, information = testing_environment.step(selected_action)

        cumulative_reward += reward
        comfort_scores.append(float(information.get("comfort_score", 0.0)))
        energy_efficiency_scores.append(float(information.get("energy_efficiency_score", 0.0)))
        colour_temperature_scores.append(float(information.get("colour_temperature_score", 0.0)))
        overall_scores.append(float(information.get("overall_score", 0.0)))

    testing_environment.close()

    return {
        "cumulative_reward": float(cumulative_reward),
        "comfort_score": float(mean(comfort_scores)),
        "energy_efficiency_score": float(mean(energy_efficiency_scores)),
        "colour_temperature_score": float(mean(colour_temperature_scores)),
        "overall_score": float(mean(overall_scores)),
    }

def train_single_run(seed):
    """Executes the full TD3 training workflow for a specific random seed."""
    run_directory = os.path.join(RESULTS_DIRECTORY, "td3", f"run_{seed}")
    best_model_directory = os.path.join(run_directory, "best_model")
    final_model_path = os.path.join(run_directory, f"td3_final_seed_{seed}")

    # Create folder structure for results and best versioning
    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(best_model_directory, exist_ok=True)

    training_environment = create_monitored_environment()
    evaluation_environment = create_monitored_environment()

    # Callback: Stop early if the agent's performance meets the requirement
    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=50,
        verbose=1,
    )

    # Callback: Periodically evaluate the model and save the best weight configuration
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=best_model_directory,
        log_path=run_directory,
        eval_freq=5000,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    # Exploration Noise: Essential for TD3 to explore the continuous action space
    action_noise = NormalActionNoise(
        mean=np.zeros(training_environment.action_space.shape[-1]),
        sigma=0.05 * np.ones(training_environment.action_space.shape[-1]),
    )

    # Initialize Twin Delayed Deep Deterministic Policy Gradient (TD3) agent
    reinforcement_learning_model = TD3(
        policy="MlpPolicy",
        env=training_environment,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,   # Size of the replay memory
        learning_starts=5000, # Initial steps used to fill the buffer before learning
        batch_size=64,        # Size of the minibatch sampled from replay buffer
        tau=0.005,            # Smoothing factor for soft updates of target networks
        gamma=0.99,           # Reward discount factor
        train_freq=1,         # Frequency of updating the model (per step)
        gradient_steps=1,     # Number of optimization steps per update
        action_noise=action_noise,
        seed=seed,
    )

    # Run the training loop
    reinforcement_learning_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=evaluation_callback,
        progress_bar=True,
    )

    # Finalize by saving model and generating summary data
    reinforcement_learning_model.save(final_model_path)
    print(f"Training finished for seed={seed}.")

    quick_test_metrics = run_quick_test(reinforcement_learning_model)

    run_summary = {
        "algorithm": "TD3",
        "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS,
        "final_model_path": f"{final_model_path}.zip",
        "best_model_directory": best_model_directory,
        "quick_test": quick_test_metrics,
    }

    # Save detailed run results to JSON for comparison
    summary_path = os.path.join(run_directory, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(run_summary, summary_file, indent=2)

    return run_summary

def main():
    """Entry point for conducting multi-run experiments and aggregating metrics."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    # Standard Gymnasium compliance check
    check_env(SmartLightingEnv(), warn=True)

    print("\n===== TD3 MULTI-RUN TRAINING =====\n")

    run_summaries = []

    for seed in SEEDS:
        print(f"\n--- TD3 Run with seed={seed} ---\n")
        run_summary = train_single_run(seed)
        run_summaries.append(run_summary)

    # Compile aggregate statistics across all seeds for reliability
    aggregate_summary = {
        "algorithm": "TD3",
        "seeds": SEEDS,
        "mean_quick_test_reward": mean(
            item["quick_test"]["cumulative_reward"] for item in run_summaries
        ),
        "mean_quick_test_overall_score": mean(
            item["quick_test"]["overall_score"] for item in run_summaries
        ),
        "runs": run_summaries,
    }

    aggregate_path = os.path.join(RESULTS_DIRECTORY, "td3", "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as aggregate_file:
        json.dump(aggregate_summary, aggregate_file, indent=2)

    print("\n===== TD3 MULTI-RUN SUMMARY =====")
    print(f"Mean quick-test overall score: {aggregate_summary['mean_quick_test_overall_score']:.2f}")

if __name__ == "__main__":
    main()