import importlib
import yaml
import os


def create_paths(paths):
    for p in paths:
        os.makedirs(p) if not os.path.exists(p) else None


def cast_values(data):
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively call the function for nested dictionaries
            data[key] = cast_values(value)
        elif isinstance(value, str):
            if value == 'None':
                data[key] = None
            else:  # Convert to float if possible (ex. 10e-3)
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
        elif isinstance(value, list):
            continue
    return data


def instantiate_from_config_with_params(config: dict, additional_params: dict):
    """
    Creates a copy of config (c), adds additional params to c['params'], and calls instantiate_from_config(c)
    """
    # config:
    #  target:
    #   ...
    #  params:
    #   ...
    c = config
    c_p = c['params']
    c_p.update(additional_params)
    c['params'].update(c_p)
    return instantiate_from_config(c)


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def get_config(path='configs/conf.yaml'):
    with open(path, 'r') as file:
        conf = yaml.safe_load(file)
    conf = cast_values(conf)
    return conf
