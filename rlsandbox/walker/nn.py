from abc import abstractmethod
from typing import Sequence, Callable, Iterable

import torch
import torch.nn.init as init
from torch import nn, Tensor


def linear_layers(
        size0: int,
        size1: int,
        *sizes: int,
        activation_builders: Sequence[type[nn.Module]],
) -> list[nn.Module]:
    sizes = [size0, size1, *sizes]

    all_layers = []
    for i in range(len(sizes) - 1):
        layers = [nn.Linear(sizes[i], sizes[i + 1])]

        if i < len(sizes) - 2:
            activations = [b() for b in activation_builders]
            layers.extend(activations)

        all_layers.append(nn.Sequential(*layers))

    return all_layers


def reinitialize_random_parameters(model: nn.Module, fraction: float) -> None:
    """
    Re-initialize a small random sample of learnable parameters of the model.

    Args:
        model (nn.Module): The model whose parameters are to be re-initialized.
        fraction (float): The fraction of parameters to re-initialize.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                num_elements = param.numel()
                num_to_reinitialize = int(fraction * num_elements)
                indices = torch.randperm(num_elements)[:num_to_reinitialize]
                flat_param = param.view(-1)
                flat_param[indices] = init.normal_(
                    torch.empty(num_to_reinitialize, device=param.device)
                )


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
