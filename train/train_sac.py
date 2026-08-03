import os
import json
from statistics import mean

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env.lighting_env import SmartLightingEnv

# Configuration for experiment results and training duration
RESULTS_DIRECTORY = "results_train"
SEEDS = [1, 2, 3]
TOTAL_TIMESTEPS = 300000

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

    # Track domain-specific scores for analysis
    comfort_scores = []
    energy_efficiency_scores = []
    colour_temperature_scores = []
    overall_scores = []

    while not (terminated or truncated):
        # Use deterministic prediction for consistent testing evaluation
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
    """Executes a full SAC training cycle for a specific random seed."""
    run_directory = os.path.join(RESULTS_DIRECTORY, "sac", f"run_{seed}")
    best_model_directory = os.path.join(run_directory, "best_model")
    final_model_path = os.path.join(run_directory, f"sac_final_seed_{seed}")

    # Initialize results directories
    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(best_model_directory, exist_ok=True)

    training_environment = create_monitored_environment()
    evaluation_environment = create_monitored_environment()

    # Callback: Stop training early if a performance threshold is met
    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=50,
        verbose=1,
    )

    # Callback: Periodically evaluate and save the highest-performing model version
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=best_model_directory,
        log_path=run_directory,
        eval_freq=5000,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    # Initialize Soft Actor-Critic (SAC) agent
    reinforcement_learning_model = SAC(
        policy="MlpPolicy",
        env=training_environment,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,   # Size of the replay buffer
        learning_starts=5000, # Steps to collect before training begins
        batch_size=128,       # Minibatch size for gradient updates
        tau=0.005,            # Soft update coefficient for target networks
        gamma=0.99,           # Discount factor
        train_freq=1,         # Update the model every step
        gradient_steps=1,     # How many gradient steps to take per update
        ent_coef=0.05,        # Entropy coefficient (fixed) to control exploration
        target_update_interval=1,
        seed=seed,
    )

    # Start the learning process
    reinforcement_learning_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=evaluation_callback,
        progress_bar=True,
    )

    # Save final model state
    reinforcement_learning_model.save(final_model_path)
    print(f"Training finished for seed={seed}.")

    # Run post-training validation
    quick_test_metrics = run_quick_test(reinforcement_learning_model)

    # Export specific run summary to JSON
    run_summary = {
        "algorithm": "SAC",
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
    """Main execution block for multi-seed SAC experiments."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    # Ensure environment compatibility with Stable Baselines3
    check_env(SmartLightingEnv(), warn=True)

    print("\n===== SAC MULTI-RUN TRAINING =====\n")

    run_summaries = []

    for seed in SEEDS:
        print(f"\n--- SAC Run with seed={seed} ---\n")
        run_summary = train_single_run(seed)
        run_summaries.append(run_summary)

    # Aggregate and save metrics across all seeds
    aggregate_summary = {
        "algorithm": "SAC",
        "seeds": SEEDS,
        "mean_quick_test_reward": mean(
            item["quick_test"]["cumulative_reward"] for item in run_summaries
        ),
        "mean_quick_test_overall_score": mean(
            item["quick_test"]["overall_score"] for item in run_summaries
        ),
        "runs": run_summaries,
    }

    aggregate_path = os.path.join(RESULTS_DIRECTORY, "sac", "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as aggregate_file:
        json.dump(aggregate_summary, aggregate_file, indent=2)

    print("\n===== SAC MULTI-RUN SUMMARY =====")
    print(f"Mean quick-test overall score: {aggregate_summary['mean_quick_test_overall_score']:.2f}")

if __name__ == "__main__":
    main()