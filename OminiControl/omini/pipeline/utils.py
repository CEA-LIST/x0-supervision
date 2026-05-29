import importlib
import torch
import numpy as np


def collate_fn(batch):
    txts = [sample['description'] for sample in batch]
    keys = set(batch[0].keys()) - {'description'}
    res = dict(description=txts)
    for key in keys:
        if isinstance(batch[0][key], np.ndarray):
            res[key] = torch.stack([torch.FloatTensor(sample[key]) for sample in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            res[key] = torch.stack([sample[key] for sample in batch])
        else:
            res[key] = [sample[key] for sample in batch]
    return res

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))