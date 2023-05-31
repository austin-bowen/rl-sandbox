from abc import abstractmethod

from rlsandbox.types import State, Action


class Agent:
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action(self, state: State) -> Action:
        ...
