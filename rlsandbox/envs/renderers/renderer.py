from abc import abstractmethod

from rlsandbox.envs.env import Env


class EnvRenderer:
    @abstractmethod
    def render(self, env: Env) -> None:
        ...
