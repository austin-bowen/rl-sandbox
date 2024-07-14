from abc import abstractmethod
from typing import Sequence, Callable, Iterable

import torch
from torch import nn, Tensor


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


class FeatureTransformer:
    def __call__(self, batch: Tensor) -> Tensor:
        return self.transform(batch)

    @abstractmethod
    def transform(self, batch: Tensor) -> Tensor:
        ...


class IdentityTransformer(FeatureTransformer):
    def transform(self, batch: Tensor) -> Tensor:
        return batch


class ConcatTransformer(FeatureTransformer):
    def __init__(
            self,
            get_additional_features: Callable[[Tensor], Iterable[Tensor]] = None,
            dim: int = 1,
    ):
        self._get_additional_features = get_additional_features
        self.dim = dim

    def transform(self, batch: Tensor) -> Tensor:
        new_features = self.get_additional_features(batch)
        new_features = (
            f.unsqueeze(self.dim) if f.dim() == 1 else f
            for f in new_features
        )

        return torch.cat((batch, *new_features), dim=self.dim)

    def get_additional_features(self, batch: Tensor) -> Iterable[Tensor]:
        if self._get_additional_features is not None:
            return self._get_additional_features(batch)

        raise NotImplementedError()
