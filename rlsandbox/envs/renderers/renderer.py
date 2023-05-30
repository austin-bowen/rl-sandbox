from abc import abstractmethod

from rlsandbox.types import State


class EnvRenderer:
    @abstractmethod
    def render(self, state: State) -> None:
        ...
