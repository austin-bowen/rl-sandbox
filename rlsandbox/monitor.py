import time
from multiprocessing import Event, Pipe, Process
from threading import Thread
from typing import Optional

from rlsandbox.agents.agent import Agent
from rlsandbox.envs.env import Env
from rlsandbox.envs.renderers.renderer import EnvRenderer
from rlsandbox.types import StateChange


class Monitor(Thread):
    env: Env
    env_renderer: EnvRenderer
    agent: Optional[Agent]

    _parent_pipe: Pipe
    _child_pipe: Pipe
    _should_stop: Event

    def __init__(self, env: Env, env_renderer: EnvRenderer, **kwargs):
        super().__init__(daemon=True, **kwargs)

        self.env = env
        self.env_renderer = env_renderer
        self.agent = None

        self._parent_pipe, self._child_pipe = Pipe()
        self._should_stop = Event()

    def __enter__(self) -> 'Monitor':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        self.join()

    def stop(self) -> None:
        self._should_stop.set()

    def set_agent(self, agent: Agent) -> None:
        self._parent_pipe.send(agent)

    def run(self) -> None:
        while not self._should_stop.is_set():
            self._update_agent()

            if self.agent is None:
                time.sleep(0.5)
                continue

            self._play_game()

    def _update_agent(self) -> None:
        if self._child_pipe.poll():
            self.agent = self._child_pipe.recv()

    def _play_game(self) -> None:
        state = self.env.reset()

        self.agent.reset()

        while True:
            self.env_renderer.render(state)

            action = self.agent.get_action(state)

            state_change: StateChange = self.env.step(action)
            state = state_change.next_state

            if state_change.done:
                break

        self.env_renderer.render(state_change.next_state)
