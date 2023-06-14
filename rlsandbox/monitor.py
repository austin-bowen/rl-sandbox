import queue
import time
from multiprocessing import Event, Queue
from threading import Thread
from typing import Optional

from rlsandbox.agents.agent import Agent
from rlsandbox.env_runner import EnvRunner
from rlsandbox.envs.env import Env
from rlsandbox.envs.renderers.renderer import EnvRenderer


class Monitor(Thread):
    env: Env
    renderer: EnvRenderer
    agent: Optional[Agent]

    _agent_queue: Queue
    _stop_event: Event

    def __init__(self, env: Env, env_renderer: EnvRenderer, **kwargs):
        super().__init__(daemon=True, **kwargs)

        self.env = env
        self.renderer = env_renderer
        self.agent = None

        self._agent_queue = Queue(1)
        self._stop_event = Event()

    def __enter__(self) -> 'Monitor':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        self.join()

    def stop(self) -> None:
        self._stop_event.set()

    def set_agent(self, agent: Agent) -> None:
        try:
            while True:
                self._agent_queue.get_nowait()
        except queue.Empty:
            pass

        self._agent_queue.put(agent)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._update_agent()

            if self.agent is None:
                time.sleep(0.5)
                continue

            runner = EnvRunner(
                self.env,
                self.agent,
                renderer=self.renderer,
                stop_event=self._stop_event,
            )

            runner.run()

    def _update_agent(self) -> None:
        try:
            self.agent = self._agent_queue.get_nowait()
        except queue.Empty:
            pass
