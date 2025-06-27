# main.py

import random
import numpy as np
import csv

from config import params
from env import SmartBuildingEnv
from agent import PPOAgent
from utils import plot_metric

seed = 42
random.seed(seed)
np.random.seed(seed)

def train():
    env = SmartBuildingEnv(params)
    states = env.reset()
    state_dim = len(states[0])
    num_agents = params["num_agents"]
    num_actions = params["num_actions"]
    agents = [PPOAgent(state_dim, num_actions, params) for _ in range(num_agents)]
    episode_rewards_list = []
    moving_avg_rewards = []
    avg_coverage_list = []
    avg_energy_list = []
    avg_cost_list = []
    anomaly_detected_list = []
    fairness_index_list = []
    avg_sensing_latency_list = []
    fl_rounds_list = []
    comm_freq_list = []
    total_federated_aggregations = 0
    total_comm_overhead = 0.0
    episodes = params["max_episodes"]
    print("Starting training...\n")
    for episode in range(episodes):
        states = env.reset()
        episode_rewards = np.zeros(num_agents)
        episode_coverage = []
        episode_energy = []
        episode_cost = []
        episode_anomalies = []
        for step in range(params["max_steps"]):
            actions = []
            for i, agent in enumerate(agents):
                a = agent.select_action(states[i])
                actions.append(a)
            next_states, rewards, done, metrics = env.step(actions)
            for i, agent in enumerate(agents):
                agent.store_reward_done(rewards[i], done)
            episode_rewards += rewards
            episode_coverage.append(metrics["coverage"])
            episode_energy.append(metrics["energy_used"])
            episode_cost.append(metrics["offload_cost"])
            episode_anomalies.append(metrics["anomaly_detected"])
            states = next_states
            if done:
                break
        # PPO update (per agent)
        for agent in agents:
            agent.finish_episode()
        # --- New Metrics: Fairness Index & Sensing Latency ---
        per_agent_coverage = np.mean(np.stack(episode_coverage), axis=0)
        numerator = (np.sum(per_agent_coverage)) ** 2
        denominator = params["num_agents"] * np.sum(per_agent_coverage ** 2)
        fairness_index = numerator / denominator if denominator > 0 else 0
        fairness_index_list.append(fairness_index)
        sensing_latencies = params["max_steps"] - env.last_sensed
        avg_sensing_latency = np.mean(sensing_latencies)
        avg_sensing_latency_list.append(avg_sensing_latency)
        # Federated aggregation
        if (episode + 1) % params["fl_agg_freq"] == 0:
            import torch
            with torch.no_grad():
                global_model_state = {}
                for key in agents[0].get_state_dict().keys():
                    global_model_state[key] = sum(agent.get_state_dict()[key] for agent in agents) / num_agents
            for agent in agents:
                agent.load_state_dict(global_model_state)
            total_federated_aggregations += 1
            total_comm_overhead += params["comm_overhead_per_agg"]
        avg_episode_reward = np.mean(episode_rewards)
        episode_rewards_list.append(avg_episode_reward)
        fl_rounds_list.append(total_federated_aggregations)
        comm_freq_list.append((episode + 1) / (total_federated_aggregations if total_federated_aggregations > 0 else 1))
        if len(episode_rewards_list) >= params["moving_avg_window"]:
            moving_avg = np.mean(episode_rewards_list[-params["moving_avg_window"]:])
        else:
            moving_avg = np.mean(episode_rewards_list)
        moving_avg_rewards.append(moving_avg)
        avg_coverage = np.mean(np.concatenate(episode_coverage))
        avg_energy = np.mean(np.concatenate(episode_energy))
        avg_cost = np.mean(np.concatenate(episode_cost))
        total_anomalies = np.sum(np.concatenate(episode_anomalies))
        avg_coverage_list.append(avg_coverage)
        avg_energy_list.append(avg_energy)
        avg_cost_list.append(avg_cost)
        anomaly_detected_list.append(total_anomalies)
        print(f"Episode {episode+1}/{episodes} completed:")
        print(f"  Avg Reward: {avg_episode_reward:.3f} | Moving Avg: {moving_avg:.3f}")
        print(f"  Avg Coverage: {avg_coverage:.2f} | Avg Energy: {avg_energy:.2f} | Avg Cost: {avg_cost:.2f}")
        print(f"  Fairness Index: {fairness_index:.3f} | Avg Sensing Latency: {avg_sensing_latency:.2f}")
        print(f"  Anomalies Detected: {total_anomalies:.1f}")
        print(f"  Federated Aggregations: {total_federated_aggregations} | Total Comm Overhead: {total_comm_overhead:.1f}\n")
    print("Training complete.")
    print("Final Avg Reward:", np.mean(episode_rewards_list))
    print("Final Moving Avg Reward:", moving_avg_rewards[-1])
