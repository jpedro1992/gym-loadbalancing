import csv
import math
import operator
from datetime import datetime
import heapq
import time
import random
from statistics import mean
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from envs.utils import normalize, EndpointRequest, get_endpoint_list, save_to_csv, calculate_gini_coefficient, \
    calculate_jains_fairness_index
import logging

# Actions - for printing purposes
ACTIONS = ["Forward-To-", "Reject"]

# Reward functions:
# NAIVE Strategy: +1 if agent accepts request, or -1 if rejects it (if resources were available)
NAIVE = 'naive'

# Multi-objective reward funtion
MULTI = 'multi'

# Latency Related
LATENCY = 'latency'  # Both Topology and Endpoint Latency

# Fairness
FAIRNESS = 'fairness'

# Cluster Types
NUM_NODE_TYPES = 5
DEFAULT_NODE_TYPES = [{"type": "edge_tier_1", "cpu": 2.0, "mem": 2.0, "cost": 1},
                      {"type": "edge_tier_2", "cpu": 2.0, "mem": 4.0, "cost": 2},
                      {"type": "fog_tier_1", "cpu": 2.0, "mem": 8.0, "cost": 4},
                      {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 8},
                      {"type": "cloud", "cpu": 8.0, "mem": 32.0, "cost": 16}]

# DEFAULTS for Env configuration
DEFAULT_NUM_EPISODE_STEPS = 100

# Topology related
DEFAULT_NUM_ENDPOINTS = 8
DEFAULT_NUM_ZONES = 4
DEFAULT_NUM_NODES = 24

# Simulation related
DEFAULT_ARRIVAL_RATE = 100
DEFAULT_CALL_DURATION = 1
DEFAULT_REWARD_FUNTION = NAIVE
DEFAULT_FILE_NAME_RESULTS = "loadbalancer_k8s_gym_results"
DEFAULT_REJECTION_ALLOWED = True

NUM_METRICS_ENDPOINT_REQUEST = 3  # Zone ID, latency threshold, departure time
NUM_METRICS_ENDPOINT = 5  # Zone ID, Node CPU of endpoint?, Avg. latency of endpoint

# Defaults for latency
# Total Latency = processing latency  + network latency
MIN_TOPOLOGY_LATENCY = 1  # corresponds to 1ms
MEDIUM_TOPOLOGY_LATENCY = 50  # corresponds to 1ms
MAX_TOPOLOGY_LATENCY = 500  # corresponds to 500ms
MIN_ENDPOINT_LATENCY = 1  # corresponds to 1ms
MAX_ENDPOINT_LATENCY = 500  # corresponds to 500ms

INCREASE_COST_PERCENTAGE = 1.7  # 10%
# INCREASE_COST_PERCENTAGE = 1.3 # 30%
# INCREASE_COST_PERCENTAGE = 1.5 # 50%
# INCREASE_COST_PERCENTAGE = 1.7 # 70%

# Defaults for CPU
MIN_CPU_USAGE = 1  # corresponds to 1%
MAX_CPU_USAGE = 100  # corresponds to 100%

# Defaults for Weights
LATENCY_WEIGHT = 0.7
CPU_WEIGHT = 0.1
GINI_WEIGHT = 0.2


class LoadBalancerK8sEnv(gym.Env):
    """ LoadBalancerK8s env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_endpoints=DEFAULT_NUM_ENDPOINTS,
                 rejection_allowed=DEFAULT_REJECTION_ALLOWED,
                 num_zones=DEFAULT_NUM_ZONES,
                 num_nodes=DEFAULT_NUM_NODES,
                 arrival_rate_r=DEFAULT_ARRIVAL_RATE,
                 call_duration_r=DEFAULT_CALL_DURATION,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS,
                 reward_function=DEFAULT_REWARD_FUNTION,
                 file_results_name=DEFAULT_FILE_NAME_RESULTS,
                 latency_weight=LATENCY_WEIGHT,
                 cpu_weight=CPU_WEIGHT,
                 gini_weight=GINI_WEIGHT):

        # Define action and observation space
        super(LoadBalancerK8sEnv, self).__init__()
        self.name = "loadbalancer_k8s_gym"
        self.__version__ = "0.0.1"
        self.reward_function = reward_function
        self.num_endpoints = num_endpoints
        self.num_zones = num_zones
        self.num_nodes = num_nodes
        self.rejection_allowed = rejection_allowed

        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[EndpointRequest] = []

        # For Latency purposes
        self.topology_latency_matrix = np.zeros((num_zones, num_zones))
        self.topology_latency_matrix_init = np.zeros((num_zones, num_zones))
        self.matrix_updated = np.zeros((num_zones, num_zones))

        self.endpoint_topology_latency = np.zeros(num_endpoints)

        self.selected_endpoint_latency = 0.0
        self.selected_endpoint_topology_latency = 0.0
        self.selected_endpoint_cpu = 1.0

        self.endpoint_latency = self.np_random.uniform(low=MIN_ENDPOINT_LATENCY, high=MAX_ENDPOINT_LATENCY,
                                                       size=num_endpoints)  # start latency between 0 and 100

        self.seed = 42
        self.np_random, seed = seeding.np_random(self.seed)

        logging.info(
            "[Init] Env: {} | "
            "Version {} | "
            "Num. Endpoints: {} | "
            "Num. Zones: {} |".format(self.name, self.__version__, num_endpoints, num_zones))

        # Defined as a matrix having as rows the nodes and columns their associated metrics
        if self.rejection_allowed:
            self.observation_space = spaces.Box(low=MIN_TOPOLOGY_LATENCY,
                                                high=MAX_TOPOLOGY_LATENCY,
                                                shape=(
                                                    num_endpoints + 1, NUM_METRICS_ENDPOINT + NUM_METRICS_ENDPOINT_REQUEST),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=MIN_TOPOLOGY_LATENCY,
                                                high=MAX_TOPOLOGY_LATENCY,
                                                shape=(num_endpoints, NUM_METRICS_ENDPOINT + NUM_METRICS_ENDPOINT_REQUEST),
                                                dtype=np.float32)

        # Default latency matrix
        # Fill the matrix with different random integers
        for z1 in range(num_zones):
            for z2 in range(num_zones):
                if z1 == z2:
                    self.topology_latency_matrix[z1][z2] = 1.0
                else:
                    value = self.np_random.integers(MIN_TOPOLOGY_LATENCY, MAX_TOPOLOGY_LATENCY)
                    self.topology_latency_matrix[z1][z2] = value
                    self.topology_latency_matrix[z2][z1] = value

        self.topology_latency_matrix_init = self.topology_latency_matrix
        self.matrix_updated = self.topology_latency_matrix_init

        logging.info("[Init] Topology Latency Matrix: {}".format(self.topology_latency_matrix))

        # Action Space
        # Balance the request to one of the available endpoints or reject it
        if self.rejection_allowed:
            self.num_actions = num_endpoints + 1
        else:
            self.num_actions = num_endpoints

        # Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)

        # Action and Observation Space
        logging.info("[Init] Action Space: {}".format(self.action_space))
        logging.info("[Init] Observation Space: {}".format(self.observation_space))
        logging.info("[Init] Observation Space Shape: {}".format(self.observation_space.shape))

        # Setting the experiment based on TeaStore deployments
        self.endpointList = get_endpoint_list()
        self.endpoint_request = None

        # New: Resource capacities based on node type
        self.node_cpu_capacity = np.zeros(num_nodes)
        self.node_zone = np.zeros(num_nodes)
        self.node_type = np.zeros(num_nodes)

        # Zone CPU capacity
        self.zone_cpu_capacity = np.zeros(num_zones)

        # Endpoint Resource Values
        self.endpoint_zone = np.zeros(num_endpoints)
        self.endpoint_zone_cpu_capacity = np.zeros(num_endpoints)
        self.endpoint_node = np.zeros(num_endpoints)
        self.endpoint_cpu_usage_percentage = np.zeros(num_endpoints)

        # logging.info("[Init] Resource Capacity calculation... ")
        for n in range(num_nodes):
            type = int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            self.node_type[n] = int(type)
            self.node_cpu_capacity[n] = DEFAULT_NODE_TYPES[type]['cpu']

            zone_id = int(self.np_random.integers(low=0, high=DEFAULT_NUM_ZONES))
            self.node_zone[n] = int(zone_id)

            self.zone_cpu_capacity[zone_id] += self.node_cpu_capacity[n]

            logging.info("[Init] node id: {} | Zone id: {} | Type: {} | cpu: {} |".format(
                n + 1,
                self.node_zone[n],
                DEFAULT_NODE_TYPES[type]['type'],
                self.node_cpu_capacity[n]))

        logging.info("[Init] Zone CPU capacity: {} |".format(self.zone_cpu_capacity))

        # Keeps track of allocated resources
        # self.node_allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=num_nodes)
        # self.node_allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=num_nodes)

        # Keeps track of Free resources
        # self.node_free_cpu = np.zeros(num_nodes)
        # self.node_free_memory = np.zeros(num_nodes)

        # Keeps track of Usage Percentage
        self.node_cpu_usage_percentage = np.zeros(num_nodes)
        # self.mem_usage_percentage = np.zeros(num_nodes)

        for n in range(num_nodes):
            self.node_cpu_usage_percentage[n] = int(self.np_random.integers(low=MIN_CPU_USAGE, high=MAX_CPU_USAGE))

        logging.info("[Init] Node Types: {}".format(self.node_type))
        logging.info("[Init] Node Zones: {}".format(self.node_zone))
        logging.info("[Init] Node CPU Usage: {}".format(self.node_cpu_usage_percentage))

        logging.info("[Init] Endpoint Node and Zone definition... ")

        self.endpoint_cpu_usage_percentage = np.zeros(num_endpoints)

        for e in range(num_endpoints):
            node = int(self.np_random.integers(low=0, high=DEFAULT_NUM_NODES))
            self.endpoint_node[e] = int(node)
            self.endpoint_zone[e] = int(self.node_zone[node])
            self.endpoint_cpu_usage_percentage[e] = self.node_cpu_usage_percentage[node]
            self.endpoint_zone_cpu_capacity[e] = self.zone_cpu_capacity[int(self.node_zone[node])]

        logging.info("[Init] Endpoint Zone: {}".format(self.endpoint_zone))
        logging.info("[Init] Endpoint Zone CPU capacity: {}".format(self.endpoint_zone_cpu_capacity))
        logging.info("[Init] Endpoint Node: {}".format(self.endpoint_node))
        logging.info("[Init] Endpoint CPU Usage: {}".format(self.endpoint_cpu_usage_percentage))

        # Variables for rewards
        self.latency_weight = latency_weight
        self.cpu_weight = cpu_weight
        self.gini_weight = gini_weight

        # Variables for logging
        self.current_step = 0
        self.current_time = 0
        self.penalty = False
        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0
        self.ep_intra_zone_prob = 0
        self.ep_inter_zone_prob = 0
        self.next_request()

        # Info & episode over
        self.total_reward = 0
        self.episode_over = False
        self.info = {}
        self.block_prob = 0
        self.ep_block_prob = 0
        self.avg_topology_latency = []
        self.avg_topology_latency_updated = []
        self.avg_endpoint_latency = []
        self.avg_cost = []
        self.avg_load_served = np.zeros(num_endpoints)
        self.avg_cpu_usage_percentage_endpoint_selected = []
        self.intra_zone_requests = 0
        self.inter_zone_requests = 0
        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = file_results_name + ".csv"
        self.obs_csv = self.name + "_obs.csv"

    # Reset Function
    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.ep_accepted_requests = 0
        self.penalty = False

        self.block_prob = 0
        self.ep_block_prob = 0
        self.ep_intra_zone_prob = 0
        self.ep_inter_zone_prob = 0
        self.avg_topology_latency = []
        self.avg_topology_latency_updated = []
        self.avg_endpoint_latency = []
        self.avg_cost = []
        self.avg_load_served = np.zeros(self.num_endpoints)
        self.avg_cpu_usage_percentage_endpoint_selected = []
        self.intra_zone_requests = 0
        self.inter_zone_requests = 0

        # Reset Deployment Data
        self.endpointList = get_endpoint_list()

        # For Latency purposes
        self.topology_latency_matrix = np.zeros((self.num_zones, self.num_zones))
        self.topology_latency_matrix_init = np.zeros((self.num_zones, self.num_zones))
        self.matrix_updated = np.zeros((self.num_zones, self.num_zones))

        self.endpoint_topology_latency = np.zeros(self.num_endpoints)
        self.selected_endpoint_latency = 0.0
        self.selected_endpoint_topology_latency = 0.0
        self.selected_endpoint_cpu = 1.0
        self.endpoint_latency = self.np_random.uniform(low=1.0, high=100.0,
                                                       size=self.num_endpoints)  # start latency between 0 and 100
        # Default values
        for z1 in range(self.num_zones):
            for z2 in range(self.num_zones):
                if z1 == z2:
                    self.topology_latency_matrix[z1][z2] = 1.0
                else:
                    value = self.np_random.integers(MIN_TOPOLOGY_LATENCY, MAX_TOPOLOGY_LATENCY)
                    self.topology_latency_matrix[z1][z2] = value
                    self.topology_latency_matrix[z2][z1] = value

        self.topology_latency_matrix_init = self.topology_latency_matrix
        self.matrix_updated = self.topology_latency_matrix_init

        # logging.info("[Reset] Resource Capacity calculation... ")
        self.node_type = [0] * self.num_nodes  # np.zeros(num_clusters)

        # Zone CPU capacity
        self.zone_cpu_capacity = np.zeros(self.num_zones)

        for n in range(self.num_nodes):
            type = int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            self.node_type[n] = type
            self.node_cpu_capacity[n] = DEFAULT_NODE_TYPES[type]['cpu']

            zone_id = int(self.np_random.integers(low=0, high=DEFAULT_NUM_ZONES))
            self.node_zone[n] = int(zone_id)
            self.zone_cpu_capacity[zone_id] += self.node_cpu_capacity[n]

            logging.info("[Reset] node id: {} | Zone id: {} | Type: {} | cpu: {} |".format(
                n + 1,
                self.node_zone[n],
                DEFAULT_NODE_TYPES[type]['type'],
                self.node_cpu_capacity[n]))

        # Keeps track of allocated resources
        # self.node_allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)
        # self.node_allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)

        # Keeps track of Usage Percentage
        self.node_cpu_usage_percentage = np.zeros(self.num_nodes)
        # self.mem_usage_percentage = np.zeros(self.num_nodes)

        for n in range(self.num_nodes):
            self.node_cpu_usage_percentage[n] = int(self.np_random.integers(low=MIN_CPU_USAGE, high=MAX_CPU_USAGE))

        logging.info("[Reset] Node Types: {}".format(self.node_type))
        logging.info("[Reset] Node Zones: {}".format(self.node_zone))
        logging.info("[Reset] Node CPU Usage: {}".format(self.node_cpu_usage_percentage))

        for e in range(self.num_endpoints):
            node = int(self.np_random.integers(low=0, high=DEFAULT_NUM_NODES))
            self.endpoint_node[e] = int(node)
            self.endpoint_zone[e] = int(self.node_zone[node])
            self.endpoint_cpu_usage_percentage[e] = float(
                "{:.2f}".format(self.node_cpu_usage_percentage[node]))

            self.endpoint_zone_cpu_capacity[e] = self.zone_cpu_capacity[int(self.node_zone[node])]

        logging.info("[Reset] Endpoint Zone CPU capacity: {}".format(self.endpoint_zone_cpu_capacity))
        logging.info("[Reset] Endpoint Zone: {}".format(self.endpoint_zone))
        logging.info("[Reset] Endpoint Node: {}".format(self.endpoint_node))
        logging.info("[Reset] Endpoint CPU Usage: {}".format(self.endpoint_cpu_usage_percentage))

        # Reset Penalty
        self.penalty = False

        # Get next request
        self.next_request()

        # return obs
        return np.array(self.get_state())

    # Step function
    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Execute one time step within the environment
        self.offered_requests += 1
        self.take_action(action)

        # Update observation before reward calculation
        # ob = self.get_state()

        # Calculate Reward
        reward = self.get_reward()
        self.total_reward += reward

        # Find correct action move for logging purposes
        move = ""
        if action < self.num_endpoints:
            move = ACTIONS[0] + "Endpoint-" + str(action + 1)
        elif action == self.num_endpoints:
            move = ACTIONS[1]

        # Logging Step and Total Reward
        logging.info('[Step {}] | Action: {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, move, reward, self.total_reward))

        # Get next request
        self.next_request()

        # Update observation
        ob = self.get_state()

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.save_obs_to_csv(self.obs_csv, np.array(ob), date)

        # episode results to save
        self.block_prob = 1 - (self.accepted_requests / self.offered_requests)
        self.ep_block_prob = 1 - (self.ep_accepted_requests / self.current_step)
        self.ep_intra_zone_prob = self.intra_zone_requests / self.current_step
        self.ep_inter_zone_prob = self.inter_zone_requests / self.current_step

        if len(self.avg_endpoint_latency) == 0 and len(self.avg_topology_latency) == 0 and len(self.avg_cost) == 0 \
                and len(self.avg_cpu_usage_percentage_endpoint_selected) == 0:
            avg_cpu = 1
            avg_c = 1
            avg_l = 1
            avg_t = 1
        else:
            avg_cpu = mean(self.avg_cpu_usage_percentage_endpoint_selected)
            avg_c = mean(self.avg_cost)
            avg_l = mean(self.avg_endpoint_latency)
            avg_t = mean(self.avg_topology_latency)

        self.info = {
            "reward_step": float("{:.2f}".format(reward)),
            "action": float("{:.2f}".format(action)),
            "reward": float("{:.2f}".format(self.total_reward)),
            "ep_block_prob": float("{:.2f}".format(self.ep_block_prob)),
            "ep_accepted_requests": float("{:.2f}".format(self.ep_accepted_requests)),
            'avg_endpoint_latency': float("{:.2f}".format(avg_l)),
            'avg_topology_latency': float("{:.2f}".format(avg_t)),
            'avg_cost': float("{:.2f}".format(avg_c)),
            'avg_cpu_endpoint_selected': float("{:.2f}".format(avg_cpu)),
            'ep_intra_zone_percentage': float("{:.2f}".format(self.ep_intra_zone_prob)),
            'ep_inter_zone_percentage': float("{:.2f}".format(self.ep_inter_zone_prob)),
            'gini': float("{:.2f}".format(calculate_gini_coefficient(self.avg_load_served))),
            'executionTime': float("{:.2f}".format(self.execution_time))
        }

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start
            # logging.info("[Step] Episode finished, saving results to csv...")

            gini = calculate_gini_coefficient(self.avg_load_served)
            # jains = calculate_jains_fairness_index(self.avg_load_served)

            '''
            logging.info("[Episode finish] Loads: {} | Gini: {} | Jains: {}".format(self.avg_load_served, gini, jains))
            logging.info("[Episode finish] Intra requests: {} | %: {} |".format(self.intra_zone_requests,
                                                                                self.ep_intra_zone_prob))
            logging.info("[Episode finish] Inter requests: {} | %: {} |".format(self.inter_zone_requests,
                                                                                self.ep_inter_zone_prob))
            '''
            save_to_csv(self.file_results, self.episode_count,
                        self.total_reward, self.ep_block_prob,
                        self.ep_accepted_requests,
                        mean(self.avg_endpoint_latency),
                        mean(self.avg_topology_latency),
                        mean(self.avg_cost),
                        mean(self.avg_cpu_usage_percentage_endpoint_selected),
                        self.ep_intra_zone_prob,
                        self.ep_inter_zone_prob,
                        gini,
                        self.execution_time)

            save_to_csv("no_cost_updated.csv", self.episode_count,
                        self.total_reward, self.ep_block_prob,
                        self.ep_accepted_requests,
                        mean(self.avg_endpoint_latency),
                        mean(self.avg_topology_latency_updated),
                        mean(self.avg_cost),
                        mean(self.avg_cpu_usage_percentage_endpoint_selected),
                        self.ep_intra_zone_prob,
                        self.ep_inter_zone_prob,
                        gini,
                        self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    # Reward Function
    def get_reward(self):
        """ Calculate Rewards """
        if self.reward_function == NAIVE:
            if self.penalty:
                # logging.info("[Get Reward] Penalty = True, penalize the agent...")
                return -1
            else:
                return 1

        # Minimize Topology Latency
        elif self.reward_function == LATENCY:
            # logging.info('[Get Reward] Minimize Latency Reward Funtion Selected...')
            total = MAX_TOPOLOGY_LATENCY + MAX_ENDPOINT_LATENCY
            if self.penalty:
                # logging.info("[Get Reward] Penalty = True (Req. Rejected), penalize agent with max latency...")
                return -total  # penalty = 0 OR -total
            else:  # agent should not be penalized...
                return -(self.selected_endpoint_latency + self.selected_endpoint_topology_latency)

        # Maximize fairness -
        elif self.reward_function == FAIRNESS:
            # logging.info('[Get Reward] Maximize Fairness Reward Funtion Selected...')
            if self.penalty:
                # logging.info("[Get Reward] Penalty = True (Req. Rejected), penalize agent with max latency...")
                return -1  # penalty = 1 s
            else:  # Gini 0 is better!
                return 1 - calculate_gini_coefficient(self.avg_load_served)

        # Multi-objective
        elif self.reward_function == MULTI:
            if self.penalty:
                return -1  # (MAX_TOPOLOGY_LATENCY + MAX_ENDPOINT_LATENCY)  # Is penalty enough?
            else:  # R=−laten1cy×e (1/cpu) × (1−fairness)
                current = self.selected_endpoint_latency + self.selected_endpoint_topology_latency
                cpu = self.selected_endpoint_cpu
                gini = calculate_gini_coefficient(self.avg_load_served)
                logging.info('[Multi Reward] latency: {} | cpu: {} | gini: {} |'.format(current, cpu, gini))

                current = normalize(current, 2 * MIN_TOPOLOGY_LATENCY, 2 * MAX_TOPOLOGY_LATENCY)
                cpu = normalize(cpu, MIN_CPU_USAGE, MAX_CPU_USAGE)
                reward = self.latency_weight * (1 - current) + self.cpu_weight * (1 - cpu) + self.gini_weight * (
                            1 - gini)

                logging.info(
                    '[Multi Reward] latency norm: {} | cpu norm: {} | gini: {} | reward: {}'.format(current, cpu, gini,
                                                                                                    reward))
                logging.info('[Multi Reward] latency part: {} | cpu part: {} | gini part: {}'.format(
                    self.latency_weight * (1 - current), self.cpu_weight * cpu, self.gini_weight * (1 - gini)))
                return reward

        else:
            logging.info('[Get Reward] Unrecognized reward: {}'.format(self.reward_function))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    # Apply the action selected by the RL agent
    def take_action(self, action):
        self.current_step += 1

        # change Network costs
        if self.current_step == 1:
            logging.info("[Take Action Update] Change network Costs...")
            logging.info("[Take Action Update] Previous: {}".format(self.matrix_updated))
            self.matrix_updated = np.zeros((self.num_zones, self.num_zones))
            for z1 in range(self.num_zones):
                for z2 in range(self.num_zones):
                    if z1 == z2:
                        self.matrix_updated[z1][z2] = 1.0
                    else:
                        value = self.topology_latency_matrix[z1][z2] * INCREASE_COST_PERCENTAGE
                        self.matrix_updated[z1][z2] = value
                        self.matrix_updated[z2][z1] = value

        logging.info("[Take Action Update] After: {}".format(self.matrix_updated))

        logging.info("[Take Action] Top. Latency Init: {} | Top. Latency Updated: {}".format(
            self.topology_latency_matrix,
            self.matrix_updated))

        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Possible Actions:
        # Selection of an Endpoint
        if action < self.num_endpoints:
            # accept request
            # logging.info("[Take Action] Balance the request...")
            self.accepted_requests += 1
            self.ep_accepted_requests += 1

            self.endpoint_request.deployed_endpoint = action
            self.endpoint_request.deployed_node = self.endpoint_node[action]
            self.endpoint_request.deployed_zone = self.endpoint_zone[action]

            # Update avg latency and cost
            input_zone = int(self.endpoint_request.input_zone)
            input_node = int(self.endpoint_request.input_node)
            output_zone = int(self.endpoint_request.deployed_zone)
            output_node = int(self.endpoint_node[action])
            output_node_type = self.node_type[output_node]
            cost = DEFAULT_NODE_TYPES[output_node_type]['cost']

            self.avg_topology_latency.append(self.topology_latency_matrix[input_zone][output_zone])
            self.avg_topology_latency_updated.append(self.matrix_updated[input_zone][output_zone])

            self.avg_endpoint_latency.append(self.endpoint_latency[action])
            self.avg_cost.append(cost)
            self.avg_load_served[action] += 1
            self.avg_cpu_usage_percentage_endpoint_selected.append(self.endpoint_cpu_usage_percentage[action])

            self.selected_endpoint_latency = self.endpoint_latency[action]
            self.selected_endpoint_topology_latency = self.topology_latency_matrix[input_zone][output_zone]
            self.selected_endpoint_cpu = self.endpoint_cpu_usage_percentage[action]

            logging.info("[Take Action] Endpoint Req - Zone: {} | Node: {} |".format(input_zone, input_node))
            logging.info("[Take Action] Endpoint Selected: {} | Zone: {} | Node: {} | Node Type: {} | Node cost: {}|"
                         .format(action, output_zone, output_node, output_node_type, cost))

            logging.info("[Take Action] Latency: {} | CPU %: {} | Init Topo. latency: {}".format(
                self.endpoint_latency[action],
                self.endpoint_cpu_usage_percentage[action],
                self.topology_latency_matrix[input_zone][output_zone]))

            logging.info("[Take Action] Latency: {} | CPU %: {} | Updated Topo. latency: {}".format(
                self.endpoint_latency[action],
                self.endpoint_cpu_usage_percentage[action],
                self.matrix_updated[input_zone][output_zone]))

            # logging.info("[Take Action] Load Served: {}".format(self.avg_load_served))

            # Check Intra or Inter Zone comm.
            # Intra same zone
            # Inter different zone
            if input_zone == output_zone:
                # logging.info("[Take Action] Intra Zone selected")
                self.intra_zone_requests += 1
            else:
                # logging.info("[Take Action] Inter Zone selected")
                self.inter_zone_requests += 1

            # logging.info("[Take Action] Cost: {}".format(DEFAULT_NODE_TYPES[output_node_type]['cost']))

            # Save expected latency and cost in deployment request
            # self.endpoint_request.expected_latency = self.topology_latency_matrix[input_zone][output_zone]
            # self.endpoint_request.expected_cost = DEFAULT_NODE_TYPES[output_node_type]['cost']

            # Enqueue Request
            self.enqueue_request(self.endpoint_request)

            # Update resources of Node hosting selected endpoint
            self.increase_resources(action, 1.15)  # 15% increase max

            # Increase Latency of selected Endpoint
            self.increase_endpoint_latency(action, 1.5)  # 50% increase max

            # Penalty is false
            self.penalty = False

        # Reject the request: give the agent a penalty
        elif action == self.num_endpoints:
            self.penalty = True
        else:
            logging.info('[Take Action] Unrecognized Action: {}'.format(action))

    def get_state(self):
        # Get Observation state
        endpoint = np.full(shape=(1, NUM_METRICS_ENDPOINT), fill_value=-1)

        observation = np.stack([self.endpoint_zone,
                                self.endpoint_zone_cpu_capacity,
                                self.endpoint_cpu_usage_percentage,
                                # self.endpoint_memory_usage_percentage,
                                self.endpoint_topology_latency,
                                self.endpoint_latency
                                ],
                               axis=1)

        logging.info('[Get State]: endpoint: {}'.format(endpoint))
        logging.info('[Get State]: endpoint shape: {}'.format(endpoint.shape))

        logging.info('[Get State]: observation: {}'.format(observation))
        logging.info('[Get State]: observation shape: {}'.format(observation.shape))

        if self.rejection_allowed:
            # Condition the elements in the set with the current request
            endpoint_request = np.tile(
                np.array(
                    [self.endpoint_request.input_zone,
                     # self.endpoint_request.input_cpu_usage, # No need?
                     # self.endpoint_request.input_memory_usage, # No need?
                     self.endpoint_request.latency_threshold,
                     self.dt,
                     ]
                ),
                (self.num_endpoints + 1, 1),
            )

            logging.info('[Get State]: endpoint_request: {}'.format(endpoint_request))
            logging.info('[Get State]: endpoint_request shape: {}'.format(endpoint_request.shape))

            observation = np.concatenate([observation, endpoint], axis=0)  # 0
            logging.info('[Get State]: obs endpoint concatenation: {}'.format(observation))
            logging.info('[Get State]: obs endpoint concatenation shape: {}'.format(observation.shape))

            observation = np.concatenate([observation, endpoint_request], axis=1)
            logging.info('[Get State]: obs endpoint_request concatenation: {}'.format(observation))
            logging.info('[Get State]: obs endpoint_request concatenation shape: {}'.format(observation.shape))

        else:
            # Condition the elements in the set with the current request
            endpoint_request = np.tile(
                np.array(
                    [self.endpoint_request.input_zone,
                     # self.endpoint_request.input_cpu_usage, # No need?
                     # self.endpoint_request.input_memory_usage, # No need?
                     self.endpoint_request.latency_threshold,
                     self.dt,
                     ]
                ),
                (self.num_endpoints, 1),
            )

            logging.info('[Get State]: endpoint_request: {}'.format(endpoint_request))
            logging.info('[Get State]: endpoint_request shape: {}'.format(endpoint_request.shape))

            # observation = np.concatenate([observation, endpoint], axis=0) #0
            # logging.info('[Get State]: obs endpoint concatenation: {}'.format(observation))
            # logging.info('[Get State]: obs endpoint concatenation shape: {}'.format(observation.shape))

            observation = np.concatenate([observation, endpoint_request], axis=1)
            logging.info('[Get State]: obs endpoint_request concatenation: {}'.format(observation))
            logging.info('[Get State]: obs endpoint_request concatenation shape: {}'.format(observation.shape))
            logging.info('[Get State]: Observation: {}'.format(observation))

        return observation

    '''
    # Save observation to csv file
    def save_obs_to_csv(self, obs_file, obs, date):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        node_obs = {}
        with file:
            fields.append('date')
            for n in range(self.num_nodes):
                fields.append("node_" + str(n + 1) + '_node_allocated_cpu')
                fields.append("node_" + str(n + 1) + '_node_cpu_capacity')
                fields.append("node_" + str(n + 1) + '_node_allocated_memory')
                fields.append("node_" + str(n + 1) + '_node_memory_capacity')
                fields.append("node_" + str(n + 1) + '_num_replicas')
                fields.append("node_" + str(n + 1) + '_cpu_request')
                fields.append("node_" + str(n + 1) + '_memory_request')
                fields.append("node_" + str(n + 1) + '_dt')

            # logging.info("[Save Obs] fields: {}".format(fields))

            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            node_obs.update({fields[0]: date})

            for n in range(self.num_nodes):
                i = self.get_iteration_number(n)
                node_obs.update({fields[i + 1]: obs[n][0]})
                node_obs.update({fields[i + 2]: obs[n][1]})
                node_obs.update({fields[i + 3]: obs[n][2]})
                node_obs.update({fields[i + 4]: obs[n][3]})
                node_obs.update({fields[i + 5]: obs[n][4]})
                node_obs.update({fields[i + 6]: obs[n][5]})
                node_obs.update({fields[i + 7]: obs[n][6]})
                node_obs.update({fields[i + 8]: obs[n][7]})
            writer.writerow(node_obs)
        return
    '''

    def get_iteration_number(self, e):
        num_fields_per_endpoint = 5
        return num_fields_per_endpoint * e

    def enqueue_request(self, request: EndpointRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    # Action masks: always valid?
    def action_masks(self):
        if self.rejection_allowed:
            valid_actions = np.ones(self.num_endpoints + 1, dtype=bool)
            for i in range(self.num_endpoints):
                valid_actions[i] = True

            # 1 additional action: Reject
            valid_actions[self.num_endpoints] = True
        else:
            valid_actions = np.ones(self.num_endpoints, dtype=bool)
            for i in range(self.num_endpoints):
                valid_actions[i] = True
        logging.info("[Action Masking] Valid actions: {} |".format(valid_actions))
        return valid_actions

    '''
    def get_low_high_scale_resources(self, prev, factor, min, max):
        low = prev
        high = float(prev * factor)

        if high > max:
            high = max
        if low < min:
            low = min

        if low == high and low != min:
            low -= 1
        if low == high and low == max:
            high += 1

        if low > high:
            low, high = high, low  # Swap values if low is greater than high

        return low, high

    def get_low_high_reduce_resources(self, prev, factor, min, max):
        low = float(prev * factor)
        high = prev

        if low < min:
            low = min
        if high > max:
            high = max

        if low == high and high != max:
            high += 1
        if low == high and high == max:
            low -= 1

        return low, high
    '''

    # Increase CPU and memory usage of Node in the episode
    def increase_resources(self, e, factor):
        hosting_node = int(self.endpoint_node[e])
        prev_cpu_usage = self.node_cpu_usage_percentage[hosting_node]

        # Ensure the new value is within the specified range
        new_cpu = max(min(prev_cpu_usage * factor, MAX_CPU_USAGE), MIN_CPU_USAGE)

        # low_cpu, high_cpu = self.get_low_high_scale_resources(prev_cpu_usage, factor, 1, 100)

        # print("[Increase Resources] prev: {} | low_cpu: {} | high_cpu: {} |".format(prev_cpu_usage, low_cpu, high_cpu))
        # self.node_cpu_usage_percentage[hosting_node] = int(self.np_random.uniform(low=low_cpu, high=high_cpu))

        self.node_cpu_usage_percentage[hosting_node] = new_cpu
        if self.node_cpu_usage_percentage[hosting_node] == 0.0:
            self.node_cpu_usage_percentage[hosting_node] = 1.0

        logging.info("[Increase Resources] Hosting Node: {} |".format(hosting_node))
        logging.info("[Increase Resources] prev_cpu: {} | "
                     "new_cpu: {} |".format(prev_cpu_usage,
                                            self.node_cpu_usage_percentage[hosting_node]))

        # logging.info("[Increase Resources] prev_cpu: {} | low_cpu: {} | "
        #             "high_cpu: {} | new_cpu: {} |".format(prev_cpu_usage, low_cpu,
        #                                                   high_cpu, self.node_cpu_usage_percentage[hosting_node]))

        # Update endpoint resource percentages
        self.endpoint_cpu_usage_percentage[e] = self.node_cpu_usage_percentage[hosting_node]

        '''
        prev_cpu = self.node_allocated_cpu[hosting_node]
        prev_mem = self.node_allocated_memory[hosting_node]
        
        low_cpu, high_cpu = self.get_low_high_scale_resources(prev_cpu, factor, 0.001,
                                                              self.node_cpu_capacity[hosting_node])
        low_mem, high_mem = self.get_low_high_scale_resources(prev_mem, factor, 0.001,
                                                              self.node_memory_capacity[hosting_node])
        
        self.node_allocated_cpu[hosting_node] = self.np_random.uniform(low=low_cpu, high=high_cpu)
        self.node_allocated_memory[hosting_node] = self.np_random.uniform(low=low_mem, high=high_mem)
        
        logging.info("[Increase Resources] Hosting Node: {} |".format(hosting_node))
        logging.info("[Increase Resources] prev_cpu: {} | low_cpu: {} | "
                     "high_cpu: {} | new_cpu: {} |".format(prev_cpu, low_cpu,
                                                           high_cpu, self.node_allocated_cpu[hosting_node]))
        logging.info("[Increase Resources] prev_mem: {} | low_mem: {} | "
                     "high_mem: {} | new_mem: {} |".format(prev_mem, low_mem,
                                                           high_mem, self.node_allocated_memory[hosting_node]))
        # Update free resources
        self.node_free_cpu[hosting_node] = self.node_cpu_capacity[hosting_node] - self.node_allocated_cpu[hosting_node]
        self.node_free_memory[hosting_node] = self.node_memory_capacity[hosting_node] - self.node_allocated_memory[
            hosting_node]

        # Update resource percentages
        self.cpu_usage_percentage[hosting_node] = self.node_allocated_cpu[hosting_node] / self.node_cpu_capacity[
            hosting_node]
        self.mem_usage_percentage[hosting_node] = self.node_allocated_memory[hosting_node] / self.node_memory_capacity[
            hosting_node]

        # Update endpoint resource percentages
        self.endpoint_cpu_usage_percentage[e] = self.cpu_usage_percentage[hosting_node]
        self.endpoint_memory_usage_percentage[e] = self.mem_usage_percentage[hosting_node]

        logging.info(
            "[Increase Resources] Hosting Node: {} | node_allocated_cpu: {} | "
            "node_allocated_mem: {} | node_free_cpu: {} | node_free_mem: {} | cpu %: {} | mem %: {}".format(
                hosting_node,
                self.node_allocated_cpu[hosting_node],
                self.node_allocated_memory[hosting_node],
                self.node_free_cpu[hosting_node],
                self.node_free_memory[hosting_node],
                self.endpoint_cpu_usage_percentage[e],
                self.endpoint_memory_usage_percentage[e]
            ))
        '''

    # Decrease CPU and memory usage of Node in the episode
    def decrease_resources(self, e, factor):
        hosting_node = int(self.endpoint_node[e])
        prev_cpu_usage = self.node_cpu_usage_percentage[hosting_node]

        # Ensure the new value is within the specified range
        new_cpu = max(min(prev_cpu_usage / factor, MAX_CPU_USAGE), MIN_CPU_USAGE)
        # low_cpu, high_cpu = self.get_low_high_reduce_resources(prev_cpu_usage, factor, 1, 100)
        # self.node_cpu_usage_percentage[hosting_node] = int(self.np_random.uniform(low=low_cpu, high=high_cpu))

        self.node_cpu_usage_percentage[hosting_node] = new_cpu
        if self.node_cpu_usage_percentage[hosting_node] == 0.0:
            self.node_cpu_usage_percentage[hosting_node] = 1.0

        logging.info("[Decrease Resources] prev_cpu: {} | "
                     "new_cpu: {} |".format(prev_cpu_usage,
                                            self.node_cpu_usage_percentage[hosting_node]))

        # logging.info("[Decrease Resources] Hosting Node: {} |".format(hosting_node))
        # logging.info("[Decrease Resources] prev_cpu: {} | low_cpu: {} | "
        #             "high_cpu: {} | new_cpu: {} |".format(prev_cpu_usage, low_cpu,
        #                                                   high_cpu, self.node_cpu_usage_percentage[hosting_node]))

        # Update endpoint resource percentages
        self.endpoint_cpu_usage_percentage[e] = self.node_cpu_usage_percentage[hosting_node]

        '''
        prev_cpu = self.node_allocated_cpu[hosting_node]
        prev_mem = self.node_allocated_memory[hosting_node]

        low_cpu, high_cpu = self.get_low_high_reduce_resources(prev_cpu, factor, 0.001,
                                                               self.node_cpu_capacity[hosting_node])
        low_mem, high_mem = self.get_low_high_reduce_resources(prev_mem, factor, 0.001,
                                                               self.node_memory_capacity[hosting_node])

        self.node_allocated_cpu[hosting_node] = self.np_random.uniform(low=low_cpu, high=high_cpu)
        self.node_allocated_memory[hosting_node] = self.np_random.uniform(low=low_mem, high=high_mem)
        
        logging.info("[Decrease Resources] Hosting Node: {} |".format(hosting_node))
        logging.info("[Decrease Resources] prev_cpu: {} | low_cpu: {} | "
                     "high_cpu: {} | new_cpu: {} |".format(prev_cpu, low_cpu,
                                                           high_cpu, self.node_allocated_cpu[hosting_node]))
        logging.info("[Decrease Resources] prev_mem: {} | low_mem: {} | "
                     "high_mem: {} | new_mem: {} |".format(prev_mem, low_mem,
                                                           high_mem, self.node_allocated_memory[hosting_node]))
        
        # Update free resources
        self.node_free_cpu[hosting_node] = self.node_cpu_capacity[hosting_node] - self.node_allocated_cpu[hosting_node]
        self.node_free_memory[hosting_node] = self.node_memory_capacity[hosting_node] - self.node_allocated_memory[
            hosting_node]

        # Update resource percentages
        self.cpu_usage_percentage[hosting_node] = self.node_allocated_cpu[hosting_node] / self.node_cpu_capacity[
            hosting_node]
        self.mem_usage_percentage[hosting_node] = self.node_allocated_memory[hosting_node] / self.node_memory_capacity[
            hosting_node]

        # Update endpoint resource percentages
        self.endpoint_cpu_usage_percentage[e] = float(
            "{:.2f}".format(self.cpu_usage_percentage[hosting_node]))
        self.endpoint_memory_usage_percentage[e] = float(
            "{:.2f}".format(self.mem_usage_percentage[hosting_node]))
        
        logging.info(
            "[Decrease Resources] Hosting Node: {} | node_allocated_cpu: {} | "
            "node_allocated_mem: {} | node_free_cpu: {} | node_free_mem: {} | cpu %: {} | mem %: {}".format(
                hosting_node,
                self.node_allocated_cpu[hosting_node],
                self.node_allocated_memory[hosting_node],
                self.node_free_cpu[hosting_node],
                self.node_free_memory[hosting_node],
                self.endpoint_cpu_usage_percentage[e],
                self.endpoint_memory_usage_percentage[e]
            ))
        '''

    # Increase endpoint latency in the episode
    def increase_endpoint_latency(self, e, factor):
        prev = int(self.endpoint_latency[e])

        prev = int(self.endpoint_latency[e])

        # Ensure the new value is within the specified range
        new_latency = max(min(prev * factor, MAX_ENDPOINT_LATENCY), MIN_ENDPOINT_LATENCY)

        self.endpoint_latency[e] = new_latency
        if self.endpoint_latency[e] == 0:
            self.endpoint_latency[e] += 1

        logging.info("[Increase Latency] endpoint: {} | prev_latency: {} "
                     "| new_latency: {}".format(e, prev, self.endpoint_latency[e]))

        '''
        low = prev
        high = int(prev * factor)

        if high > MAX_ENDPOINT_LATENCY:
            high = MAX_ENDPOINT_LATENCY
        if low < MIN_ENDPOINT_LATENCY:
            low = MIN_ENDPOINT_LATENCY
        if low == high and low != MIN_ENDPOINT_LATENCY:
            low -= 1
        if low == high and low == MIN_ENDPOINT_LATENCY:
            high += 1

        self.endpoint_latency[e] = self.np_random.integers(low=low, high=high)

        if self.endpoint_latency[e] == 0:
            self.endpoint_latency[e] += 1
        
        logging.info("[Increase Latency] endpoint: {} | prev_latency: {} "
                     "| low: {} | high: {} | new_latency: {}".format(e, prev,
                                                                     low, high, self.endpoint_latency[e]))
        '''

    # Decrease Latency in the episode
    def decrease_endpoint_latency(self, e, factor):
        prev = int(self.endpoint_latency[e])

        # Ensure the new value is within the specified range
        new_latency = max(min(prev / factor, MAX_ENDPOINT_LATENCY), MIN_ENDPOINT_LATENCY)

        self.endpoint_latency[e] = new_latency
        if self.endpoint_latency[e] == 0:
            self.endpoint_latency[e] += 1

        logging.info("[Decrease Latency] endpoint: {} | prev_latency: {} "
                     "| new_latency: {}".format(e, prev, self.endpoint_latency[e]))

        '''
        low = int(prev * factor)
        high = prev

        if low < MIN_ENDPOINT_LATENCY:
            low = MIN_ENDPOINT_LATENCY
        if high > MAX_ENDPOINT_LATENCY:
            high = MAX_ENDPOINT_LATENCY
        if low == high and high != MAX_ENDPOINT_LATENCY:
            high += 1
        if low == high and high == MAX_ENDPOINT_LATENCY:
            low -= 1

        self.endpoint_latency[e] = self.np_random.integers(low=low, high=high)

        if self.endpoint_latency[e] == 0:
            self.endpoint_latency[e] += 1

        
        logging.info("[Decrease Latency] endpoint: {} | prev_latency: {} "
                     "| low: {} | high: {} | new_latency: {}".format(e, prev,
                                                                     low, high, self.endpoint_latency[e]))
        '''

    # Remove deployment request
    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        # logging.info("[Dequeue] Request {}...".format(deployment_request))
        # logging.info("[Dequeue] Request will be terminated...")
        # logging.info("[Dequeue] Before: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.node_allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.node_free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.node_allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.node_free_memory))

        e = deployment_request.deployed_endpoint
        # Update cpu and memory of node hosting endpoint
        self.decrease_resources(e, 1.15)  # 15% max reduction

        # Decrease Latency
        self.decrease_endpoint_latency(e, 1.15)  # 15% max reduction

        # logging.info("[Dequeue] After: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.node_allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.node_free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.node_allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.node_free_memory))

    # Create a deployment request
    def deployment_generator(self):
        endpoint_list = get_endpoint_list()
        random = self.np_random.integers(low=0, high=len(endpoint_list))
        e = endpoint_list[random - 1]

        # Simulate Node of Endpoint
        e.input_node = self.np_random.integers(low=0, high=self.num_nodes)

        # Get Zone of Endpoint
        e.input_zone = int(self.node_zone[e.input_node])

        # Get CPU and memory usage percentage from hosting node
        e.input_cpu_usage = self.node_cpu_usage_percentage[e.input_node]
        # e.input_memory_usage = self.mem_usage_percentage[e.input_node]
        return e

    # Select (random) the next deployment request
    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.endpoint_request = self.deployment_generator()

        '''
        logging.info('[Next Request]: Name: {} | '
                     'Zone: {} | '
                     'Hosting Node: {} | '
                     'CPU: {} | '
                     'MEM: {}'.format(self.endpoint_request.name,
                                      self.endpoint_request.input_zone,
                                      self.endpoint_request.input_node,
                                      self.endpoint_request.input_cpu_usage,
                                      self.endpoint_request.input_memory_usage))
        '''

        # Update Endpoint Latency based on Endpoint Request Zone -> Agent does not learn of in the obs space

        for e in range(self.num_endpoints):
            z = int(self.endpoint_zone[e])
            self.endpoint_topology_latency[e] = self.topology_latency_matrix[z][self.endpoint_request.input_zone]

        logging.info('[Next Request]: Updated Endpoint Latency Matrix: {} |'.format(self.endpoint_topology_latency))
