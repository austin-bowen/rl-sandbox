from typing import Sequence

from torch import nn


def linear_layers(
        size0: int,
        size1: int,
        *sizes: int,
        activation_builders: Sequence[type[nn.Module]],
) -> list[nn.Module]:
    sizes = [size0, size1, *sizes]

    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        if i < len(sizes) - 2:
            activations = [b() for b in activation_builders]
            layers.extend(activations)

    return layers
