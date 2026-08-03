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
| Algorithm | Full Name                                       |
|-----------|-------------------------------------------------|
|    TD3    | Twin Delayed Deep Deterministic Policy Gradient |
|    A2C    | Advantage Actor-Critic                          |
|    PPO    | Proximal Policy Optimisation                    |
|    SAC    | Soft Actor-Critic                               |

Each algorithm was evaluated on training stability, sample efficiency, and suitability for the continuous action space required by this environment.

## Repository Contents
- `G4_report.docx` — Full project report, covering problem framing, theoretical background, environment design, algorithm selection and justification, training methodology, results, and conclusions
- Source code for the environment, RL agents, training, and evaluation (to be added)

## Module Information
- **Module:** BMDS2114 Machine Learning
- **Semester:** 202601
- **Programme:** RDSY2S3
- **Tutorial Group:** 2
- **Tutor:** Dr. Lim Siew Mooi

## Team
| No. | Name           | Registration No. |
|-----|----------------|------------------|
|  1  | Ngoh Jia Ying  |    24WMR08011    |
|  2  | Lee Cheng Chee |    24WMR07994    |
|  3  | Ooi Ying Wei   |    24WMR08016    |
|  4  | Tim Kam        |    24WMR08038    |
