__all__ = [
    'Env',
    'GymWrapper',
]

from abc import abstractmethod
from typing import Protocol

from rlsandbox.base import State, Action, StateChange


class Env(Protocol):
    @abstractmethod
    def get_state(self) -> State:
        ...

    @abstractmethod
    def reset(self) -> State:
        ...

    @abstractmethod
    def step(self, action: Action) -> StateChange:
        ...


class GymWrapper(Env):
    def __init__(self, wrapped_env):
        self.wrapped_env = wrapped_env

        self._state = None

    def get_state(self) -> State:
        return self._state

    def reset(self) -> State:
        self._state = self.wrapped_env.reset()
        return self._state

    def step(self, action: Action) -> StateChange:
        state = self._state
        next_state, reward, done, truncated, info = self.wrapped_env.step(action)
        self._state = next_state

        return StateChange(state, action, reward, next_state, done or truncated)
