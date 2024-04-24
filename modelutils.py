import torch
import torch.nn as nn

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
EMPTY_CACHE = torch.cuda.empty_cache if torch.cuda.is_available() else lambda: None
SYNCHRONIZE = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None

DEFAULT_Q_LAYERS = [nn.Conv2d, nn.Linear]

def find_layers(module, layers: list = DEFAULT_Q_LAYERS, name='', terminal=[]):
    """
    Retrieve {name: module} pairings for all layers where type(module) in layers.
    Specify `terminal` to indicate layers that should be retrieved but whose
    children should not be explored.
    """
    res = {}
    if type(module) in layers+terminal or (len(layers) == 0 and len(list(module.named_children())) == 0):
        res.update({name: module})
    if type(module) not in terminal:
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1, terminal=terminal
            ))
    return res
