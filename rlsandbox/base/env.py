__all__ = [
    'Env',
    'GymWrapper',
    'StateTransformer',
    'RewardTransformer',
    'EnvTransformer',
]

from abc import abstractmethod
from typing import Protocol, Callable

from rlsandbox.base.sar import State, Action, StateChange, Reward


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


StateTransformer = Callable[[State], State]
RewardTransformer = Callable[[StateChange], Reward]


class EnvTransformer(Env):
    def __init__(
            self,
            wrapped_env: Env,
            state_transformer: StateTransformer = None,
            reward_transformer: RewardTransformer = None,
    ):
        self.wrapped_env = wrapped_env
        self.state_transformer = state_transformer
        self.reward_transformer = reward_transformer

    def reset(self) -> State:
        state = self.wrapped_env.reset()
        return self.transform_state(state)

    def get_state(self) -> State:
        state = self.wrapped_env.get_state()
        return self.transform_state(state)

    def step(self, action: Action) -> StateChange:
        state_change = self.wrapped_env.step(action)

        state_change.state = self.transform_state(state_change.state)
        state_change.next_state = self.transform_state(state_change.next_state)
        state_change.reward = self.transform_reward(state_change)

        return state_change

    def transform_state(self, state: State) -> State:
        return state if self.state_transformer is None else self.state_transformer(state)

    def transform_reward(self, state_change: StateChange) -> Reward:
        return state_change.reward if self.reward_transformer is None else self.reward_transformer(state_change)
