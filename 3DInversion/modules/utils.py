import json
import pickle
import yaml
import math

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_yaml(yaml_path):
    with open(yaml_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_pkl(pkl_path):
    with open(pkl_path) as f:
        return pickle.load(f)

def save_json(json_path, file):
    with open(json_path, 'w') as f:
        json.dump(file, f)

def save_yaml(yaml_path, file):
    with open(yaml_path, 'w') as f:
        yaml.dump(file, f)

def save_pkl(pkl_path, file):
    with open(pkl_path, 'w') as f:
        pickle.dump(file, f)

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp