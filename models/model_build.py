import warnings

from torch import nn


def build_norm(norm_layer):
    if norm_layer == 'nn.LayerNorm':
        return nn.LayerNorm
    elif isinstance(norm_layer, nn.Module):
        return norm_layer
    else:
        warnings.warn(f'norm_layer:{norm_layer} is not supported yet! '
                      f'this string will be used directly. ')


def build_act(act):
    if act == 'nn.GELU':
        return nn.GELU
    elif isinstance(act, nn.Module):
        return act
    else:
        warnings.warn(f'activation function:{act} is not supported yet! '
                      f'this string will be used directly. ')