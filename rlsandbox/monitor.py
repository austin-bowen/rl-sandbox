import queue
import time
from multiprocessing import Event, Queue
from threading import Thread
from typing import Optional, Literal

from rlsandbox.base import Agent, Env
from rlsandbox.base.renderer import EnvRenderer
from rlsandbox.env_runner import EnvRunner


class Monitor(Thread):
    env: Env
    renderer: EnvRenderer
    update_agent_on: Literal['done', 'step']
    agent: Optional[Agent]

    _agent_queue: Queue
    _stop_event: Event

    def __init__(
            self,
            env: Env,
            env_renderer: EnvRenderer,
            update_agent_on: Literal['done', 'step'] = 'done',
            **kwargs,
    ):
        if update_agent_on not in ['done', 'step']:
            raise ValueError(
                f'update_agent_on must be one of "done" or "step"; got {update_agent_on}'
            )

        super().__init__(daemon=True, **kwargs)

        self.env = env
        self.renderer = env_renderer
        self.update_agent_on = update_agent_on
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
            )

            state = self.env.reset()

            while not state.done and not self._stop_event.is_set():
                state_change = runner.step(state)
                state = state_change.next_state

                if self.update_agent_on == 'step':
                    self._update_agent()
                    runner.agent = self.agent

    def _update_agent(self) -> None:
        try:
            self.agent = self._agent_queue.get_nowait()
            self.agent.reset()
        except queue.Empty:
            pass
