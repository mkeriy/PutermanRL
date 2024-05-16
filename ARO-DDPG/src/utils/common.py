import numpy as np
import json


def get_config(file_path: str):
    with open(file_path) as json_data:
        data = json.load(json_data)
    return data


def return_state_dim(spec):
    result = 0
    for i, k in enumerate(spec):
        result += spec[k].shape[0]
    return result


def process_state(obs):
    result = None
    for i, k in enumerate(obs):
        if result is None:
            result = obs[k]
        else:
            result = np.concatenate([result, obs[k]])
    return result