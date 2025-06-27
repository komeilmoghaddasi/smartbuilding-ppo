# env.py

import numpy as np
from config import params, n_sensing, n_local_ratio

class SmartBuildingEnv:
    def __init__(self, params):
        self.num_agents = params["num_agents"]
        self.num_zones = params["num_zones_per_agent"]
        self.time_slot = params["time_slot"]
        self.max_steps = params["max_steps"]
        self.battery_capacity = params["battery_capacity"]
        self.sensing_power_high = params["sensing_power_high"]
        self.sensing_power_medium = params["sensing_power_medium"]
        self.sensing_power_low = params["sensing_power_low"]
        self.harvested_energy_mean = params["harvested_energy_mean"]
        self.local_cpu_capacity = params["local_cpu_capacity"]
        self.cloud_cost_per_unit = params["cloud_cost_per_unit"]
        self.compute_demand_mean = params["compute_demand_mean"]
        self.sensing_levels = params["sensing_levels"]
        self.local_compute_ratios = params["local_compute_ratios"]
        self.anomaly_chance = params["anomaly_chance"]
        self.reset()

    def reset(self):
        self.timestep = 0
        self.batteries = np.ones(self.num_agents) * self.battery_capacity
        self.last_sensed = np.zeros((self.num_agents, self.num_zones))
        self.coverage_ok = np.ones((self.num_agents, self.num_zones))
        self.anomalies = (np.random.rand(self.max_steps, self.num_agents, self.num_zones) < self.anomaly_chance).astype(float)
        return self._get_states()

    def step(self, actions):
        self.timestep += 1
        rewards = np.zeros(self.num_agents)
        coverage_rates = np.zeros(self.num_agents)
        battery_usage = np.zeros(self.num_agents)
        offload_cost = np.zeros(self.num_agents)
        anomaly_detected = np.zeros(self.num_agents)
        next_states = []
        done = (self.timestep >= self.max_steps)
        for agent_id, action in enumerate(actions):
            sensing_part = action // n_local_ratio
            local_ratio_idx = action % n_local_ratio
            sensing_idxs = []
            rem = sensing_part
            for _ in range(self.num_zones):
                sensing_idxs.append(rem % n_sensing)
                rem = rem // n_sensing
            sensing_idxs = sensing_idxs[::-1]
            sensing_rates = [self.sensing_levels[idx] for idx in sensing_idxs]
            sensing_periods = [8, 4, 2, 1]
            local_ratio = params["local_compute_ratios"][local_ratio_idx]
            harvested = np.random.normal(self.harvested_energy_mean, 0.1)
            harvested = max(harvested, 0.0)
            energy_used = 0
            new_sensed = 0
            for z, rate in enumerate(sensing_rates):
                if rate == 0:
                    self.coverage_ok[agent_id, z] = 0
                else:
                    period = sensing_periods[rate]
                    if self.timestep % period == 0:
                        energy = [0, self.sensing_power_low, self.sensing_power_medium, self.sensing_power_high][rate]
                        energy_used += energy
                        self.coverage_ok[agent_id, z] = 1
                        self.last_sensed[agent_id, z] = self.timestep
                        new_sensed += 1
            self.batteries[agent_id] = min(self.battery_capacity, self.battery_capacity if self.batteries[agent_id] + harvested - energy_used > self.battery_capacity else self.batteries[agent_id] + harvested - energy_used)
            battery_usage[agent_id] = energy_used
            coverage = np.mean(self.coverage_ok[agent_id, :])
            coverage_rates[agent_id] = coverage
            compute_demand = max(0.0, np.random.normal(self.compute_demand_mean, 1.0))
            local_compute = min(local_ratio * compute_demand, self.local_cpu_capacity)
            cloud_compute = compute_demand - local_compute
            cost = cloud_compute * self.cloud_cost_per_unit
            offload_cost[agent_id] = cost
            detected = 0
            if self.timestep < self.anomalies.shape[0]:
                for z in range(self.num_zones):
                    if sensing_rates[z] > 0 and self.anomalies[self.timestep, agent_id, z] > 0.5:
                        detected += 1
            anomaly_detected[agent_id] = detected
            rewards[agent_id] = (
                2.0 * coverage
                + 1.0 * detected
                - 0.5 * energy_used
                - 1.5 * cost
            )
            next_states.append(self._get_state(agent_id))
        metrics = {
            "coverage": coverage_rates,
            "energy_used": battery_usage,
            "offload_cost": offload_cost,
            "anomaly_detected": anomaly_detected
        }
        return np.array(next_states), rewards, done, metrics

    def _get_state(self, agent_id):
        state = [self.batteries[agent_id] / self.battery_capacity]
        state += list((self.timestep - self.last_sensed[agent_id, :]) / self.max_steps)
        state += list(self.coverage_ok[agent_id, :])
        return np.array(state)

    def _get_states(self):
        return np.array([self._get_state(i) for i in range(self.num_agents)])
