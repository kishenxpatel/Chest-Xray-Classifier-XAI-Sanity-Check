import copy
import torch
import torch.nn as nn

# Ordered (name, module) pairs from output (classifier) toward input (stem)
def get_top_down_layers(model):
    layers = []
    if hasattr(model, 'classifier'):
        layers.append(('classifier', model.classifier))
    block_order = ['denseblock4', 'transition3', 'denseblock3', 'transition2',
                   'denseblock2', 'transition1', 'denseblock1']
    for name in block_order:
        if hasattr(model.features, name):
            layers.append((name, getattr(model.features, name)))
    for name in ['norm5', 'pool0', 'relu0', 'norm0', 'conv0']:
        if hasattr(model.features, name):
            layers.append((name, getattr(model.features, name)))
    return layers


def _reinitialise_module(module, seed):
    torch.manual_seed(seed)
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()

# Copy of model with the top 'depth' layers (output-to-input) randomised
# depth=0 -> unchanged (fully trained)
# depth=len(layers) -> fully randomised
def cascade_randomise(model, depth: int, seed: int):
    model_copy = copy.deepcopy(model)
    layers = get_top_down_layers(model_copy)
    for i in range(depth):
        _, module = layers[i]
        _reinitialise_module(module, seed=seed + i)
    return model_copy