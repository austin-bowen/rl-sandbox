__all__ = [
    'Env',
    'GymWrapper',
    'StateTransformer',
    'ActionTransformer',
    'StateChangeTransformer',
    'EnvTransformer',
]

from abc import abstractmethod
from typing import Protocol, Callable

from rlsandbox.base.sar import State, Action, StateChange


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
        state, info = self.wrapped_env.reset()
        self._state = state
        return state

    def step(self, action: Action) -> StateChange:
        state = self._state
        next_state, reward, done, truncated, info = self.wrapped_env.step(action)
        self._state = next_state

        return StateChange(state, action, reward, next_state, done or truncated)


StateTransformer = Callable[[State], State]
ActionTransformer = Callable[[Action], Action]
StateChangeTransformer = Callable[[StateChange], StateChange]


class EnvTransformer(Env):
    def __init__(
            self,
            wrapped_env: Env,
            state_transformer: StateTransformer = None,
            action_transformer: ActionTransformer = None,
            state_change_transformer: StateChangeTransformer = None,
    ):
        self.wrapped_env = wrapped_env
        self.state_transformer = state_transformer
        self.action_transformer = action_transformer
        self.state_change_transformer = state_change_transformer

    def reset(self) -> State:
        state = self.wrapped_env.reset()
        return self.transform_state(state)

    def get_state(self) -> State:
        state = self.wrapped_env.get_state()
        return self.transform_state(state)

    def step(self, action: Action) -> StateChange:
        action = self.transform_action(action)
        state_change = self.wrapped_env.step(action)

        state_change.state = self.transform_state(state_change.state)
        state_change.next_state = self.transform_state(state_change.next_state)

        return self.transform_state_change(state_change)

    def transform_state(self, state: State) -> State:
        return state if self.state_transformer is None else self.state_transformer(state)

    def transform_action(self, action: Action) -> Action:
        return action if self.action_transformer is None else self.action_transformer(action)

    def transform_state_change(self, state_change: StateChange) -> StateChange:
        return (
            state_change if self.state_change_transformer is None
            else self.state_change_transformer(state_change)
        )
