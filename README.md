# RL Adaptive Smart Lighting
Reinforcement Learning-Based Adaptive Smart Lighting for Human Comfort and Energy Efficiency

## Overview
Modern smart buildings aim to optimize both human comfort and energy efficiency, and lighting plays a critical role in this — directly influencing visual comfort, alertness, and operational costs. Conventional lighting systems typically rely on fixed schedules or simple rule-based thresholds, which struggle to adapt to dynamic conditions such as fluctuating occupancy, changing daylight levels, and individual comfort preferences.

This project addresses that gap by framing lighting control as a continuous-control Reinforcement Learning (RL) problem. An RL agent learns to monitor ambient conditions and dynamically adjust brightness and Correlated Colour Temperature (CCT) in real time to maximize both energy efficiency and human comfort, including support for circadian rhythm alignment.

## Objectives
- Design a Markov Decision Process (MDP) that realistically models an adaptive lighting environment
- Implement and compare multiple continuous-control RL algorithms for this task
- Train, tune, and evaluate the best-performing agent
- Analyze trade-offs between energy efficiency and human comfort

## Methodology
The project follows a structured RL development pipeline:
1. **Environment Definition** — Custom MDP with a defined state space, continuous action space (brightness and CCT), and a reward function balancing comfort and energy efficiency
2. **Environment Simulation** — A multi-episode simulation environment modeling realistic lighting scenarios and transition dynamics
3. **Algorithm Selection** — Implementation and comparison of four actor-critic algorithms suited to continuous action spaces
4. **Model Exploration** — Experimentation with different exploration strategies and hyperparameter configurations
5. **Fine-Tuning** — Hyperparameter optimization, learning stabilization, and generalization testing
6. **Training & Evaluation** — Final agent training and evaluation against defined performance metrics

## Algorithms Compared
| Algorithm | Full Name |
|---|---|
| TD3 | Twin Delayed Deep Deterministic Policy Gradient |
| A2C | Advantage Actor-Critic |
| PPO | Proximal Policy Optimisation |
| SAC | Soft Actor-Critic |

Each algorithm was evaluated on training stability, sample efficiency, and suitability for the continuous action space required by this environment.

## Repository Contents
| File / Folder | Description |
|---|---|
| `G4_report.docx` | Full project report — problem framing, theoretical background, environment design, algorithm selection and justification, training methodology, results, and conclusions |
| `G4_video (version 2).mp4` | Project demo/presentation video |
| `lighting_env.py` | Custom lighting environment (MDP: state space, action space, reward structure, transition dynamics) |
| `exploration.ipynb` | Notebook for exploring exploration strategies |
| `final_coding.ipynb` | Main notebook for agent implementation and training |
| `model_comparison.ipynb` | Notebook comparing TD3, A2C, PPO, and SAC performance |
| `app.py` | Application script (interface/demo) |
| `index.html` | Web front-end for the app |
| `episode_data.json` | Recorded episode data from training/simulation |
| `train/` | Training scripts and related files |
| `tune/` | Hyperparameter tuning scripts and related files |
| `results_train/` | Training results and outputs |

## Module Information
- **Module:** BMDS2114 Machine Learning
- **Semester:** 202601

## Team
- Ngoh Jia Ying
- Lee Cheng Chee
- Ooi Ying Wei
- Tim Kam
