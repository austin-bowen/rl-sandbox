from abc import abstractmethod

from rlsandbox.base import State, StateChange


class EnvRenderer:
    @abstractmethod
    def render_state(self, state: State) -> None:
        ...

    @abstractmethod
    def render_state_change(self, state_change: StateChange) -> None:
        ...
