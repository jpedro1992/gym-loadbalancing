import logging

import numpy as np
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm
from envs.loadbalancer_k8s_env import LoadBalancerK8sEnv
from envs.dqn_deepset import DQN_DeepSets
from envs.ppo_deepset import PPO_DeepSets

SEED = 42
env_kwargs = {"n_nodes": 10, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_deepset_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

# Logging
logging.basicConfig(filename='run_test.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

LOW_NODES = 24
HIGH_NODES = 48

LOW_ZONES = 4
HIGH_ZONES = 16

NUM_NODES = 48
NUM_ZONES = 12

# Defaults for Weights -> change here to run
LATENCY_WEIGHT = 1.0
CPU_WEIGHT = 0.0
GINI_WEIGHT = 0.0

if __name__ == "__main__":
    # Define here variables for testing
    num_endpoints = [
        6]  # [6, 12, 16, 24, 32, 48, 64, 80, 128, 150, 180]  # - 150 is 25 times higher and 180 is 30 times higher
    reward_function = 'multi'
    alg = 'ppo'
    path = "latency-aware"  # 'fairness-aware' #'latency-aware' # 'lat_0.5_cpu_0_gini_0.5'
    random, seed = seeding.np_random(SEED)
    n_episodes = 2000  # 100
    rejection = True

    i = 0
    for e in num_endpoints:
        num_nodes = NUM_NODES
        num_zones = NUM_ZONES

        print("Initiating run for {} with: endpoints: {} | zones: {}| nodes: {} |".format(alg, e, num_zones, num_nodes))

        env = LoadBalancerK8sEnv(num_nodes=num_nodes, num_zones=num_zones, num_endpoints=e,
                                 rejection_allowed=rejection,
                                 arrival_rate_r=100,
                                 call_duration_r=1, episode_length=100,
                                 reward_function=reward_function,
                                 latency_weight=LATENCY_WEIGHT, cpu_weight=CPU_WEIGHT, gini_weight=GINI_WEIGHT,
                                 file_results_name=str(i) + '_loadbalancer_gym_results_num_endpoints_' + str(e))
        env.reset()
        _, _, _, info = env.step(0)
        info_keywords = tuple(info.keys())

        envs = DummyVecEnv([lambda: LoadBalancerK8sEnv(
            num_nodes=num_nodes, num_zones=num_zones, num_endpoints=e,
            rejection_allowed=rejection,
            arrival_rate_r=100,
            call_duration_r=1, episode_length=100,
            reward_function=reward_function,
            latency_weight=LATENCY_WEIGHT, cpu_weight=CPU_WEIGHT, gini_weight=GINI_WEIGHT,
            file_results_name=str(i) + '_loadbalancer_gym_results_num_endpoints_' + str(e))])

        envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

        # PPO or DQN
        agent = None
        if alg == "ppo":
            agent = PPO_DeepSets(envs, seed=SEED, tensorboard_log=None)
        elif alg == 'dqn':
            agent = DQN_DeepSets(envs, seed=SEED, tensorboard_log=None)
        else:
            print('Invalid algorithm!')

        # Adapt the path accordingly
        agent.load(f"./results/loadbalancer/" + reward_function + "/" + path + "/"
                   + alg + "_deepsets_env_loadbalancer_num_endpoints_6_num_zones_4_reward_"
                   + reward_function + "_totalSteps_200000_run_1/"
                   + alg + "_deepsets_env_loadbalancer_num_endpoints_6_num_zones_4_reward_"
                   + reward_function + "_totalSteps_200000")

        # Test the agent for n_episodes
        for _ in tqdm(range(n_episodes)):
            obs = envs.reset()
            action_mask = np.array(envs.env_method("action_masks"))
            done = False
            while not done:
                action = agent.predict(obs, action_mask)
                obs, reward, dones, info = envs.step(action)
                action_mask = np.array(envs.env_method("action_masks"))
                done = dones[0]

        i += 1
