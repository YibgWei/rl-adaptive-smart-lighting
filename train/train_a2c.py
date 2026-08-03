import os
import json
from statistics import mean

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env.lighting_env import SmartLightingEnv

# Configuration for results and training duration
RESULTS_DIRECTORY = "results_train"
SEEDS = [1, 2, 3]
TOTAL_TIMESTEPS = 500000

def create_monitored_environment():
    """Wraps the environment in a Monitor to record training statistics."""
    return Monitor(SmartLightingEnv())

def run_quick_test(model):
    """Executes a single test episode using the trained model to gather final metrics."""
    testing_environment = SmartLightingEnv()
    observation, information = testing_environment.reset(seed=0)

    cumulative_reward = 0.0
    terminated = False
    truncated = False

    # Metric tracking lists
    comfort_scores = []
    energy_efficiency_scores = []
    colour_temperature_scores = []
    overall_scores = []

    while not (terminated or truncated):
        # Use deterministic prediction for consistent testing results
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
    """Handles the full training workflow for a specific random seed."""
    run_directory = os.path.join(RESULTS_DIRECTORY, "a2c", f"run_{seed}")
    best_model_directory = os.path.join(run_directory, "best_model")
    final_model_path = os.path.join(run_directory, f"a2c_final_seed_{seed}")

    # Create directories for results and best models only
    os.makedirs(run_directory, exist_ok=True)
    os.makedirs(best_model_directory, exist_ok=True)

    training_environment = create_monitored_environment()
    evaluation_environment = create_monitored_environment()

    # Early stopping if agent hits target performance
    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=50,
        verbose=1,
    )

    # Periodic evaluation and automatic saving of the best performing model
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=best_model_directory,
        log_path=run_directory,
        eval_freq=5000,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    # Initialize A2C model without TensorBoard logging
    reinforcement_learning_model = A2C(
        policy="MlpPolicy",
        env=training_environment,
        verbose=1,
        learning_rate=3e-4,
        n_steps=32,
        gamma=0.995,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,  # No tensorboard_log argument here
    )

    # Begin the learning process
    reinforcement_learning_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=evaluation_callback,
        progress_bar=True,
    )

    # Final save of the model weights
    reinforcement_learning_model.save(final_model_path)
    print(f"Training finished for seed={seed}. Final model saved.")

    # Generate summary metrics
    quick_test_metrics = run_quick_test(reinforcement_learning_model)

    run_summary = {
        "algorithm": "A2C",
        "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS,
        "final_model_path": f"{final_model_path}.zip",
        "best_model_directory": best_model_directory,
        "quick_test": quick_test_metrics,
    }

    # Save summary to JSON
    summary_path = os.path.join(run_directory, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(run_summary, summary_file, indent=2)

    return run_summary

def main():
    """Orchestrates the multi-run experiment and aggregates data."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    # Validate that the environment is compatible with SB3
    check_env(SmartLightingEnv(), warn=True)

    print("\n===== A2C MULTI-RUN TRAINING (NO TENSORBOARD) =====\n")

    run_summaries = []

    for seed in SEEDS:
        print(f"\n--- A2C Run with seed={seed} ---\n")
        run_summary = train_single_run(seed)
        run_summaries.append(run_summary)

    # Compile final aggregate metrics
    aggregate_summary = {
        "algorithm": "A2C",
        "seeds": SEEDS,
        "mean_quick_test_reward": mean(
            item["quick_test"]["cumulative_reward"] for item in run_summaries
        ),
        "mean_quick_test_overall_score": mean(
            item["quick_test"]["overall_score"] for item in run_summaries
        ),
        "runs": run_summaries,
    }

    # Save final aggregate result
    aggregate_path = os.path.join(RESULTS_DIRECTORY, "a2c", "aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as aggregate_file:
        json.dump(aggregate_summary, aggregate_file, indent=2)

    print("\n===== A2C MULTI-RUN SUMMARY =====")
    print(f"Mean quick-test overall score: {aggregate_summary['mean_quick_test_overall_score']:.2f}")

if __name__ == "__main__":
    main()