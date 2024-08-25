import heapq
from dataclasses import dataclass
from typing import Collection

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rlsandbox.base import StateChange


@dataclass(slots=True)
class ExtendedStateChange(StateChange):
    reward_sum: float = None
    loss: float = None


class StateChangeDataset(Dataset):
    device: torch.device
    data: list[ExtendedStateChange]

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        device = self.device
        return (
            torch.tensor(item.state, device=device),
            torch.tensor(item.action, device=device, dtype=torch.int64),
            torch.tensor(item.reward, device=device),
            torch.tensor(item.next_state, device=device),
            torch.tensor(item.done, device=device),
        )


class OnlyStateAndRewardSumDataset(Dataset):
    def __init__(self, wrapped_dataset: StateChangeDataset):
        super().__init__()
        self.wrapped_dataset = wrapped_dataset

    def __len__(self) -> int:
        return len(self.wrapped_dataset)

    def __getitem__(self, idx: int):
        item = self.wrapped_dataset.data[idx]
        device = self.wrapped_dataset.device
        return (
            torch.tensor(item.state, device=device),
            torch.tensor(item.reward_sum, device=device),
        )


class MostRecentDataset(StateChangeDataset):
    max_len: int

    def __init__(self, max_len: int, device: torch.device):
        super().__init__(device)
        self.max_len = max_len

    def extend(self, items: Collection[ExtendedStateChange]) -> None:
        self.data.extend(items)

        if len(self) > self.max_len:
            self.data = self.data[-self.max_len:]


class RandomDropDataset(StateChangeDataset):
    max_len: int
    rng: np.random.Generator

    def __init__(
            self,
            max_len: int,
            device: torch.device,
            rng: np.random.Generator = None,
    ):
        super().__init__(device)
        self.max_len = max_len
        self.rng = rng or np.random.default_rng()

    def extend(self, items: Collection[ExtendedStateChange]) -> None:
        item_count_to_drop = len(self) + len(items) - self.max_len

        if item_count_to_drop <= 0:
            self.data.extend(items)
            return

        indexes_to_drop = set(self.rng.choice(
            range(len(self)),
            size=item_count_to_drop,
            replace=False,
        ))

        self.data = [
            item for i, item in enumerate(self.data)
            if i not in indexes_to_drop
        ]
        self.data.extend(items)

        assert len(self) <= self.max_len, (len(self), self.max_len)


class LossWeightedRandomDropDataset(StateChangeDataset):
    max_len: int
    rng: np.random.Generator

    def __init__(
            self,
            max_len: int,
            device: torch.device,
            rng: np.random.Generator = None,
    ):
        super().__init__(device)
        self.max_len = max_len
        self.rng = rng or np.random.default_rng()

    def extend(self, items: Collection[ExtendedStateChange]) -> None:
        item_count_to_drop = len(self) + len(items) - self.max_len

        if item_count_to_drop <= 0:
            self.data.extend(items)
            return

        keep_logits = torch.tensor([item.loss for item in self.data], device=self.device)
        # TODO: Test this line
        keep_logits = keep_logits / keep_logits.std()
        keep_probs = F.softmax(keep_logits, dim=0).cpu().numpy()

        indexes_to_keep = set(self.rng.choice(
            range(len(self)),
            size=len(self) - item_count_to_drop,
            replace=False,
            p=keep_probs,
        ))

        self.data = [
            item for i, item in enumerate(self.data)
            if i in indexes_to_keep
        ]
        self.data.extend(items)

        assert len(self) <= self.max_len, (len(self), self.max_len)


class HighestLossDataset(StateChangeDataset):
    max_len: int
    rng: np.random.Generator

    def __init__(
            self,
            max_len: int,
            device: torch.device,
            rng: np.random.Generator = None,
    ):
        super().__init__(device)
        self.max_len = max_len
        self.rng = rng or np.random.default_rng()

    def extend(self, items: Collection[ExtendedStateChange]) -> None:
        item_count_to_drop = len(self) + len(items) - self.max_len

        if item_count_to_drop <= 0:
            self.data.extend(items)
            return

        self.data = heapq.nlargest(
            n=len(self) - item_count_to_drop,
            iterable=self.data,
            key=lambda item: item.loss,
        )

        self.data.extend(items)

        assert len(self) <= self.max_len, (len(self), self.max_len)
