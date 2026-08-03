import os
import json
from statistics import mean

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env.lighting_env import SmartLightingEnv

# Configuration for experiment results
RESULTS_DIRECTORY = "results_train"
SEEDS = [1, 2, 3]
TOTAL_TIMESTEPS = 500000

def create_monitored_environment():
    """Wraps the environment in a Monitor to record episode statistics."""
    return Monitor(SmartLightingEnv())

def run_quick_test(model):
    """Evaluates the trained agent on a single test episode to extract performance metrics."""
    testing_environment = SmartLightingEnv()
    observation, information = testing_environment.reset(seed=0)

    cumulative_reward = 0.0
    terminated = False
    truncated = False

    # Track domain-specific scores
    comfort_scores = []
    energy_efficiency_scores = []
    colour_temperature_scores = []
    overall_scores = []

    while not (terminated or truncated):
        # Deterministic prediction for consistent testing evaluation
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
    """Executes a full PPO training cycle for a specific random seed."""
    run_directory = os.path.join(RESULTS_DIRECTORY, "ppo", f"run_{seed}")
    best_model_directory = os.path.join(run_directory, "best_model")
    final_model_path = os.path.join(run_directory, f"ppo_final_seed_{seed}")

    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(best_model_directory, exist_ok=True)

    training_environment = create_monitored_environment()
    evaluation_environment = create_monitored_environment()

    # Callback: Stop early if the agent reaches a specific performance target
    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=50,
        verbose=1,
    )

    # Callback: Periodically evaluate the agent and save the best performing weights
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=best_model_directory,
        log_path=run_directory,
        eval_freq=5000,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    # Initialize Proximal Policy Optimization (PPO) agent
    reinforcement_learning_model = PPO(
        policy="MlpPolicy",
        env=training_environment,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,          # Steps to run for each env per update
        batch_size=64,        # Minibatch size for gradient descent
        n_epochs=10,          # Number of passes when optimizing the surrogate loss
        gamma=0.99,           # Discount factor
        gae_lambda=0.95,      # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,       # Clipping parameter for PPO
        ent_coef=0.01,        # Entropy coefficient to encourage exploration
        vf_coef=0.5,          # Value function coefficient
        max_grad_norm=0.5,    # Gradient clipping limit
        seed=seed,
    )

    # Start the training process
    reinforcement_learning_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=evaluation_callback,
        progress_bar=True,
    )

    # Save the final trained model
    reinforcement_learning_model.save(final_model_path)
    print(f"Training finished for seed={seed}.")

    # Run post-training validation
    quick_test_metrics = run_quick_test(reinforcement_learning_model)

    # Log results to a summary file
    run_summary = {
        "algorithm": "PPO",
        "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS,
        "final_model_path": f"{final_model_path}.zip",
        "best_model_directory": best_model_directory,
        "quick_test": quick_test_metrics,
    }

    summary_path = os.path.join(run_directory, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(run_summary, summary_file, indent=2)

    return run_summary

def main():
    """Main execution entry point for multi-seed PPO training."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    # Standard check to ensure the environment matches Gymnasium API
    check_env(SmartLightingEnv(), warn=True)

    print("\n===== PPO MULTI-RUN TRAINING =====\n")

    run_summaries = []

    for seed in SEEDS:
        print(f"\n--- PPO Run with seed={seed} ---\n")
        run_summary = train_single_run(seed)
        run_summaries.append(run_summary)

    # Calculate aggregate performance across all seeds
    aggregate_summary = {
        "algorithm": "PPO",
        "seeds": SEEDS,
        "mean_quick_test_reward": mean(
            item["quick_test"]["cumulative_reward"] for item in run_summaries
        ),
        "mean_quick_test_overall_score": mean(
            item["quick_test"]["overall_score"] for item in run_summaries
        ),
        "runs": run_summaries,
    }

    aggregate_path = os.path.join(RESULTS_DIRECTORY, "ppo", "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as aggregate_file:
        json.dump(aggregate_summary, aggregate_file, indent=2)

    print("\n===== PPO MULTI-RUN SUMMARY =====")
    print(f"Mean quick-test overall score: {aggregate_summary['mean_quick_test_overall_score']:.2f}")

if __name__ == "__main__":
    main()