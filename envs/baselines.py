import numpy.typing as npt
import gym
import numpy as np


def topology_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible endpoint that minimizes the topology latency."""
    feasible_endpoints = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[TOPO-Greedy] Feasible endpoints: {}".format(feasible_endpoints))

    if len(feasible_endpoints) == 0:
        return len(action_mask) - 1
    return feasible_endpoints[np.argmin(env.endpoint_topology_latency[feasible_endpoints])]


def zone_cpu_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible endpoint that maximizes the cpu capacity of zones."""
    feasible_endpoints = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Zone-CPU-Greedy] Feasible endpoints: {}".format(feasible_endpoints))

    # Get the endpoint with the highest CPU capacity
    if len(feasible_endpoints) == 0:
        return len(action_mask) - 1
    return feasible_endpoints[np.argmax(env.endpoint_zone_cpu_capacity[feasible_endpoints])]


def endpoint_cpu_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible endpoint that minimizes the cpu usage of the endpoint."""
    feasible_endpoints = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Endpoint-CPU-Greedy] Feasible endpoints: {}".format(feasible_endpoints))

    # Get the endpoint with the lowest CPU capacity
    if len(feasible_endpoints) == 0:
        return len(action_mask) - 1
    return feasible_endpoints[np.argmin(env.endpoint_cpu_usage_percentage[feasible_endpoints])]