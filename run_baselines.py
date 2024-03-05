import logging

import numpy as np
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.loadbalancer_k8s_env import LoadBalancerK8sEnv
from envs.baselines import topology_greedy_policy, zone_cpu_greedy_policy, endpoint_cpu_greedy_policy

MONITOR_PATH = "./results/greedy_monitor.csv"

# Logging
logging.basicConfig(filename='run_baselines.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

NUM_NODES = 48
NUM_ZONES = 12

# Defaults for Weights -> change here to run
LATENCY_WEIGHT = 1.0
CPU_WEIGHT = 0.0
GINI_WEIGHT = 0.0

TOPOLOGY_GREEDY = 'topo'
ZONE_CPU_GREEDY = 'zone_cpu'
ENDPOINT_CPU = 'endpoint_cpu'

if __name__ == "__main__":
    policy = TOPOLOGY_GREEDY
    num_nodes = NUM_NODES
    num_zones = NUM_ZONES
    num_endpoints = [6] # 12, 16, 24, 32, 48, 64, 80, 128, 150, 180]
    n_episodes = 2000
    rejection = True

    i = 0
    for e in num_endpoints:
        print("Initiating run for {} with: endpoints: {} | zones: {}| nodes: {} |".format(policy, e, num_zones, num_nodes))

        env = LoadBalancerK8sEnv(num_nodes=num_nodes, num_zones=num_zones, num_endpoints=e,
                                 arrival_rate_r=100, call_duration_r=1,
                                 rejection_allowed=rejection,
                                 episode_length=100,
                                 reward_function='naive',
                                 latency_weight=LATENCY_WEIGHT, cpu_weight=CPU_WEIGHT, gini_weight=GINI_WEIGHT,
                                 file_results_name=str(i) + "_" + policy + '_baselines_num_endpoints_' + str(e))
        env.reset()
        _, _, _, info = env.step(0)
        info_keywords = tuple(info.keys())
        env = LoadBalancerK8sEnv(num_nodes=num_nodes, num_zones=num_zones, num_endpoints=e,
                                 arrival_rate_r=100, call_duration_r=1,
                                 rejection_allowed=rejection,
                                 episode_length=100,
                                 reward_function='naive',
                                 latency_weight=LATENCY_WEIGHT, cpu_weight=CPU_WEIGHT, gini_weight=GINI_WEIGHT,
                                 file_results_name=str(i) + "_" + policy + '_baselines_num_endpoints_' + str(e))

        # env = Monitor(env, filename=MONITOR_PATH, info_keywords=info_keywords)
        returns = []
        for _ in tqdm(range(n_episodes)):
            obs = env.reset()
            action_mask = env.action_masks()
            return_ = 0.0
            done = False
            while not done:
                if policy == TOPOLOGY_GREEDY:
                    action = topology_greedy_policy(env, action_mask)
                elif policy == ZONE_CPU_GREEDY:
                    action = zone_cpu_greedy_policy(env, action_mask)
                elif policy == ENDPOINT_CPU:
                    action = endpoint_cpu_greedy_policy(env, action_mask)
                else:
                    print("unrecognized policy!")

                obs, reward, done, info = env.step(action)
                action_mask = env.action_masks()
                return_ += reward
            returns.append(return_)

        i += 1