import os
import csv
import json
from itertools import product
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.lighting_env import SmartLightingEnv

# Tuning Experiment Configuration
RESULTS_DIR = "results_tune/tuning_ppo"
TOTAL_TIMESTEPS = 200000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
FINAL_TEST_EPISODES = 5

# Hyperparameter Search Grid for PPO (8 combinations)
TUNING_VALUES = {
    "learning_rate": [1e-4, 3e-4],
    "batch_size": [64, 128],
    "n_steps": [256, 512],
}

def make_env():
    """Utility to create a monitored instance of the environment."""
    return Monitor(SmartLightingEnv())

def evaluate_model(model):
    """Conducts a multi-episode final evaluation to ensure metric stability."""
    rewards, overall = [], []

    for ep in range(FINAL_TEST_EPISODES):
        env = SmartLightingEnv()
        obs, _ = env.reset(seed=ep)

        done, trunc = False, False
        total_reward = 0.0
        overall_scores = []

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            total_reward += reward
            overall_scores.append(info["overall_score"])

        rewards.append(total_reward)
        overall.append(mean(overall_scores))
        env.close()

    return mean(rewards), mean(overall)

def train_one(label, config):
    """Executes a single training run for a specific PPO configuration."""
    path = os.path.join(RESULTS_DIR, label)
    os.makedirs(path, exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    # Callback to track the best model version during the tuning process
    callback = EvalCallback(
        eval_env,
        best_model_save_path=path,
        log_path=path,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    # Initialize PPO with grid configuration
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0, # Keeps terminal clean during batch processing
    )

    model.learn(TOTAL_TIMESTEPS, callback=callback)

    # Validate final performance
    reward, overall = evaluate_model(model)

    # Extract learning curves from the evaluations file
    eval_file = os.path.join(path, "evaluations.npz")
    if os.path.exists(eval_file):
        data = np.load(eval_file)
        timesteps = data["timesteps"]
        rewards_curve = data["results"].mean(axis=1)
    else:
        timesteps, rewards_curve = [], []

    train_env.close()
    eval_env.close()

    return {
        "label": label,
        "config": config,
        "reward": reward,
        "overall": overall,
        "timesteps": timesteps.tolist() if hasattr(timesteps, "tolist") else timesteps,
        "curve": rewards_curve.tolist() if hasattr(rewards_curve, "tolist") else rewards_curve,
    }

def plot_top_curves(results):
    """Plots learning curves for the top 3 configurations for visual comparison."""
    plt.figure(figsize=(10, 6))

    top = sorted(results, key=lambda x: x["overall"], reverse=True)[:3]

    for i, r in enumerate(top):
        if len(r["timesteps"]) > 0:
            plt.plot(r["timesteps"], r["curve"], label=f"{r['label']} (Overall: {r['overall']:.2f})")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("PPO Learning Curves: Top 3 Configurations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ppo_top_eval_curves.png"))
    plt.close()

def main():
    """Main execution loop for PPO hyperparameter grid search."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    # Generate Cartesian product of all tuning hyperparameters
    combos = list(product(
        TUNING_VALUES["learning_rate"],
        TUNING_VALUES["batch_size"],
        TUNING_VALUES["n_steps"]
    ))

    for i, (lr, bs, ns) in enumerate(combos):
        label = f"C{i+1}"
        config = {
            "learning_rate": lr,
            "batch_size": bs,
            "n_steps": ns,
        }

        print(f"Training Configuration {label}/{len(combos)}: {config}")
        res = train_one(label, config)
        results.append(res)

    # Save summary results to CSV
    with open(os.path.join(RESULTS_DIR, "ppo_8comb_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "LR", "Batch", "N_Steps", "Reward", "Overall"])
        for r in results:
            writer.writerow([
                r["label"],
                r["config"]["learning_rate"],
                r["config"]["batch_size"],
                r["config"]["n_steps"],
                r["reward"],
                r["overall"]
            ])

    # Save the best configuration to JSON
    best = max(results, key=lambda x: x["overall"])
    with open(os.path.join(RESULTS_DIR, "best_ppo_combination.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Bar chart for overall performance comparison
    plt.figure(figsize=(10, 6))
    plt.bar([r["label"] for r in results], [r["overall"] for r in results], color='salmon')
    plt.title("PPO Overall Performance Comparison")
    plt.ylabel("Overall Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ppo_overall.png"))
    plt.close()

    plot_top_curves(results)
    print(f"PPO Tuning complete. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()