__all__ = [
    'State',
    'Action',
    'Reward',
    'S',
    'A',
    'StateChange',
]

from dataclasses import dataclass
from typing import Any, TypeVar, Generic

State = Any
Action = Any
Reward = float

S = TypeVar('S', bound=State)
A = TypeVar('A', bound=Action)


@dataclass
class StateChange(Generic[S, A]):
    state: S
    action: A
    reward: Reward
    next_state: S
    done: bool
