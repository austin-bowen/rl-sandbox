from typing import Iterable

import torch
from torch import nn, Tensor

from rlsandbox.base.utils import zip_require_same_len


def parallel_eval_on_input(models: Iterable[nn.Module], input: Tensor) -> list[Tensor]:
    return [model(input) for model in models]
    futures = [torch.jit.fork(model, input) for model in models]
    return [torch.jit.wait(future) for future in futures]


def parallel_eval_on_inputs(models: Iterable[nn.Module], inputs: Iterable[Tensor]) -> list[Tensor]:
    futures = [
        torch.jit.fork(model, input)
        for model, input in zip_require_same_len(models, inputs)
    ]

    return [torch.jit.wait(future) for future in futures]
