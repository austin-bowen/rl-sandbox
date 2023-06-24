from dataclasses import dataclass
from typing import Protocol, Sequence

from rlsandbox.agents.agent import Agent
from rlsandbox.envs.env import Env
from rlsandbox.envs.renderers.renderer import EnvRenderer
from rlsandbox.types import State, StateChange


class Event(Protocol):
    def is_set(self) -> bool:
        ...


@dataclass
class EnvRunner:
    env: Env
    agent: Agent
    renderer: EnvRenderer = None
    stop_event: Event = None

    def run(self) -> Sequence[StateChange]:
        state_changes = []
        state = self.reset()

        while not state.done and not self._should_stop():
            state_change = self.step(state)
            state_changes.append(state_change)
            state = state_change.next_state

        return state_changes

    def reset(self) -> State:
        self.agent.reset()
        state = self.env.reset()
        self._render(state=state)
        return state

    def step(self, state: State) -> StateChange:
        action = self.agent.get_action(state)
        state_change = self.env.step(action)
        self._render(state_change=state_change)
        return state_change

    def _render(self, state: State = None, state_change: StateChange = None) -> None:
        if self.renderer is not None:
            if state is not None:
                self.renderer.render_state(state)
            elif state_change is not None:
                self.renderer.render_state_change(state_change)
            else:
                raise RuntimeError('Must provide either state or state_change')

    def _should_stop(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()
