import csv
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# DeploymentRequest Info
@dataclass
class EndpointRequest:
    name: str
    arrival_time: float
    latency_threshold: int  # Latency threshold that should be respected
    departure_time: float

    input_zone: int = None  # zone id of endpoint
    input_node: int = None  # node id of endpoint
    input_cpu_usage: float = None  # node hosting endpoint
    # input_memory_usage: float = None  # node hosting endpoint

    deployed_endpoint: int = None  # Node of endpoint selected
    deployed_node: int = None  # Node of endpoint selected
    deployed_zone: int = None  # Zone of endpoint selected
    expected_latency: int = None  # expected latency after balancing
    expected_cost: int = None  # expected cost after balancing


# Reverses a dict
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


# Endpoint List based on TeaStore application
def get_endpoint_list():
    endpoint_list = [
        # 1 webui
        EndpointRequest(name="teastore-webui",
                        arrival_time=0, departure_time=0,
                        latency_threshold=400),
        # 2 registry
        EndpointRequest(name="teastore-registry",
                        arrival_time=0, departure_time=0,
                        latency_threshold=200),
        # 3 image
        EndpointRequest(name="teastore-image",
                        arrival_time=0, departure_time=0,
                        latency_threshold=150),
        # 4 auth
        EndpointRequest(name="teastore-auth",
                        arrival_time=0, departure_time=0,
                        latency_threshold=250),
        # 5 persistence
        EndpointRequest(name="teastore-persistence",
                        arrival_time=0, departure_time=0,
                        latency_threshold=450),
        # 6 db
        EndpointRequest(name="teastore-db",
                        arrival_time=0, departure_time=0,
                        latency_threshold=375),
        # 7 recommender
        EndpointRequest(name="teastore-recommender",
                        arrival_time=0, departure_time=0,
                        latency_threshold=500),
    ]
    return endpoint_list


# TODO: modify function
'''
def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )
'''


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0  # Avoid division by zero if min_value equals max_value
    return (value - min_value) / (max_value - min_value)


def save_to_csv(file_name, episode, reward, ep_block_prob, ep_accepted_requests,
                avg_endpoint_latency, avg_topology_latency, avg_cost, avg_cpu_endpoint_selected,
                ep_intra_zone_percentage, ep_inter_zone_percentage,
                gini, execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')

    with file:
        fields = ['episode', 'reward', 'ep_block_prob', 'ep_accepted_requests', 'avg_endpoint_latency',
                  'avg_topology_latency', 'avg_cost',
                  'avg_cpu_endpoint_selected', 'ep_intra_zone_percentage', 'ep_inter_zone_percentage',
                  'gini', 'execution_time']

        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'reward': float("{:.2f}".format(reward)),
             'ep_block_prob': float("{:.2f}".format(ep_block_prob)),
             'ep_accepted_requests': float("{:.2f}".format(ep_accepted_requests)),
             'avg_endpoint_latency': float("{:.2f}".format(avg_endpoint_latency)),
             'avg_topology_latency': float("{:.2f}".format(avg_topology_latency)),
             'avg_cost': float("{:.2f}".format(avg_cost)),
             'avg_cpu_endpoint_selected': float("{:.2f}".format(avg_cpu_endpoint_selected)),
             'ep_intra_zone_percentage': float("{:.2f}".format(ep_intra_zone_percentage)),
             'ep_inter_zone_percentage': float("{:.2f}".format(ep_inter_zone_percentage)),
             'gini': float("{:.2f}".format(gini)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )


# Calculation of Gini Coefficient
# 0 is better - 1 is worse!
def calculate_gini_coefficient(loads):
    n = len(loads)
    total_load = sum(loads)
    mean_load = total_load / n if n != 0 else 0

    if mean_load == 0:
        return 0  # Handle the case where all loads are zero to avoid division by zero

    gini_numerator = sum(abs(loads[i] - loads[j]) for i in range(n) for j in range(n))
    gini_coefficient = gini_numerator / (2 * n ** 2 * mean_load)

    return gini_coefficient


# Calculation of Jain's Fairness Index
# 0 is worse - 1 is better!
def calculate_jains_fairness_index(loads):
    n = len(loads)

    if n == 0:
        return 0  # Handle the case where there are no resources to avoid division by zero

    total_load = sum(loads)
    jains_numerator = total_load ** 2
    jains_denominator = n * sum(load_i ** 2 for load_i in loads)

    jains_fairness_index = jains_numerator / jains_denominator if jains_denominator != 0 else 0

    return jains_fairness_index
