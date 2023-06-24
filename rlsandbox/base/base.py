from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, Generic

Action = Any
Reward = float


class State(Protocol):
    done: bool


S = TypeVar('S', bound=State)
A = TypeVar('A', bound=Action)


@dataclass
class StateChange(Generic[S, A]):
    state: S
    action: A
    reward: Reward
    next_state: S

    @property
    def done(self) -> bool:
        return self.next_state.done


class Agent:
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state: State) -> Action:
        ...


class Env:
    @abstractmethod
    def get_state(self) -> State:
        ...

    @abstractmethod
    def reset(self) -> State:
        ...

    @abstractmethod
    def step(self, action: Action) -> StateChange:
        ...
