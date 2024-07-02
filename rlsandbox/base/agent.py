__all__ = [
    'Agent',
]

from abc import abstractmethod

from rlsandbox.base import State, Action


class Agent:
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state: State) -> Action:
        ...
