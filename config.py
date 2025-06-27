# config.py
params = {
    "num_agents": 10,
    "num_zones_per_agent": 5,
    "max_steps": 30,
    "time_slot": 1.0,
    "battery_capacity": 100.0,
    "sensing_power_high": 1.5,
    "sensing_power_medium": 0.8,
    "sensing_power_low": 0.3,
    "harvested_energy_mean": 0.5,
    "local_cpu_capacity": 10.0,
    "cloud_cost_per_unit": 0.1,
    "compute_demand_mean": 8.0,
    "anomaly_chance": 0.15,
    "sensing_levels": [0, 1, 2, 3],
    "local_compute_ratios": [0.0, 0.5, 1.0],
    "max_episodes": 500,
    "gamma_discount": 0.98,
    "ppo_clip_eps": 0.2,
    "ppo_epochs": 5,
    "ppo_batch_size": 128,
    "gae_lambda": 0.95,
    "learning_rate": 2e-4,
    "buffer_capacity": 3000,
    "moving_avg_window": 5,
    "fl_agg_freq": 10,
    "comm_overhead_per_agg": 2.0,
}

n_sensing = len(params["sensing_levels"])
n_local_ratio = len(params["local_compute_ratios"])
n_zones = params["num_zones_per_agent"]
params["num_actions"] = (n_sensing ** n_zones) * n_local_ratio
