from abc import abstractmethod

from rlsandbox.types import Action, StateChange, State


class Env:
    @abstractmethod
    def reset(self) -> State:
        ...

    @abstractmethod
    def step(self, action: Action) -> StateChange:
        ...
