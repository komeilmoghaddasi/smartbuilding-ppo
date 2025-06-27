# SmartBuilding Federated PPO 

A modular, extensible research codebase for **Federated Multi-Agent Reinforcement Learning** in smart building environments, using Proximal Policy Optimization (PPO) and federated learning.


## Table of Contents

1. [ Features](#-features)
2. [📂 Repository Layout](#-repository-layout)
3. [ Quick Start](#-quick-start)
4. [🔧 Configuration](#-configuration)
5. [ Outputs & Logging](#-outputs--logging)
6. [ Extending the Codebase](#-extending-the-codebase)
7. [🤝 Contributing](#-contributing)
8. [📜 Citation](#-citation)
9. [🙌 Acknowledgements](#-acknowledgements)



## ✨ Features

* **Custom Smart-Building Simulator** with zone-level sensing, energy harvesting, cloud offload cost, and anomaly events.
* **Multi-Agent PPO** (actor–critic) with Generalized Advantage Estimation.
* **Federated Learning** — periodic on-device model aggregation without sharing raw data.
* **Rich Metrics**: reward, coverage, energy, offload cost, anomalies detected, Jain fairness, sensing latency, comm overhead.
* **CSV Logging + Pretty Plots** for instant analysis.
* **Pure Python 3.8+**, minimal dependencies, no proprietary frameworks.



## 📂 Repository Layout

```
smartbuilding-ppo/
├── agent.py          # PPO networks + training logic
├── config.py         # All hyper-parameters & env settings
├── env.py            # SmartBuildingEnv gym-like environment
├── main.py           # Training script & CLI entry-point
├── utils.py          # Plotting & helper functions
│
├── requirements.txt  # Python package list
├── .gitignore        # Ignore cache, logs, etc.
└── README.md         # You are here ✔
```



## ⚡ Quick Start

### 1 · Clone

```bash
git clone https://github.com/yourusername/smartbuilding-ppo.git
cd smartbuilding-ppo
```

### 2 · (Optional) Create a virtual env

```bash
python -m venv venv
source venv/bin/activate           # macOS/Linux
venv\Scripts\activate              # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Train

```bash
python main.py
```

The script prints live progress and, on completion, writes `smartbuilding_metrics_ppo_full.csv` and opens plots for all tracked metrics.



## 🔧 Configuration

All knobs live in **`config.py`**:

```python
params = {
    "num_agents": 10,
    "num_zones_per_agent": 5,
    "max_steps": 30,
    ...
    "fl_agg_freq": 10,              # ↔ federated round frequency
    "comm_overhead_per_agg": 2.0    # ↔ MB cost per aggregation
}
```

Simply edit values or load overrides in your own scripts.  Every param is documented inline.



## 📊 Outputs & Logging

| Artifact    | Path                                 | Description                                 |
| ----------- | ------------------------------------ | ------------------------------------------- |
| **CSV log** | `smartbuilding_metrics_ppo_full.csv` | All metrics per episode.                    |
| **Plots**   | displayed interactively              | Reward curves, coverage, energy, cost, etc. |
| **Console** | stdout                               | Compact training summary every episode.     |



## 🧩 Extending the Codebase

| Want to …                  | Start here                                                |
| -------------------------- | --------------------------------------------------------- |
| Add another RL algorithm   | `agent.py` (implement new Agent class)                    |
| Tweak environment dynamics | `env.py`                                                  |
| Log to TensorBoard         | Modify `utils.py` / integrate `torch.utils.tensorboard`   |
| Run multiple experiments   | Write a shell or Python sweep script (e.g. with 🪄 Hydra) |

Because every module is standalone, you can import them into Jupyter notebooks for rapid prototyping.


## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!


## 📜 Citation

The paper of this study is currently under review.


## 🙌 Acknowledgements

Studied by Komeil Moghaddasi (k.moghaddasi@ieee.org). Inspired by advances in federated RL and smart-building analytics.

<p align="center"><i>Feel free to star ⭐ the repo if it helps you!</i></p>
