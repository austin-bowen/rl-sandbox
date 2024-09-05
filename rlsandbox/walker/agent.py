import heapq
import random
from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np
import torch
from torch import Tensor

from rlsandbox.base.utils import zip_require_same_len
from rlsandbox.walker.logging import log_metrics
from rlsandbox.walker.model import WalkerWorldModel, WorldModelInput, ValueModelInput, WorldModelOutput

Action = np.ndarray
"""4-dimensional array with range [-1, 1]."""


class Stats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.lookahead_depth = []
        self.reward_value_ratio = []
        self.random_action_prob = []

    def emit(self, step: int = None) -> None:
        metrics = dict()

        if self.lookahead_depth:
            metrics['agent_lookahead_depth'] = np.mean(self.lookahead_depth)

        if self.reward_value_ratio:
            metrics['agent_reward_value_ratio'] = np.mean(self.reward_value_ratio)

        if self.random_action_prob:
            metrics['agent_random_action_prob'] = np.mean(self.random_action_prob)

        if metrics:
            log_metrics(metrics, step=step)

        self.reset()


class Agent:
    def reset(self) -> None:
        pass

    def get_action(self, state: Tensor) -> Action:
        raise NotImplementedError

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class WalkerAgent(Agent):
    def __init__(
            self,
            world_model: WalkerWorldModel,
            value_model,
    ):
        self.world_model = world_model
        self.value_model = value_model
        self.stats = Stats()
        self.action_history = []

    def reset(self) -> None:
        self.action_history.clear()

    def get_action(
            self,
            state: Tensor,
            unstick_after: int = 10,
    ) -> Action:
        action, reward = self.rollout(state)

        self.action_history.append(action)

        should_choose_random_action = self._last_action_count() > unstick_after
        if should_choose_random_action:
            return self._random_action()
        else:
            return action

    def _last_action_count(self) -> int:
        last_action = self.action_history[-1]
        count = 0

        for action in reversed(self.action_history):
            if not np.allclose(action, last_action):
                break

            count += 1

        return count

    def _random_action(self) -> Action:
        return np.random.randint(low=0, high=2, size=4) * 2 - 1

    def eval(self) -> None:
        self.world_model.eval()

        try:
            self.value_model.eval()
        except AttributeError:
            pass

    def train(self) -> None:
        self.world_model.train()

        try:
            self.value_model.train()
        except AttributeError:
            pass

    def rollout(self, state: Tensor) -> tuple[Action, float]:
        # return self.multi_rollout(
        #     state,
        #     rollout_func=self.rollout_beam_search,
        # )

        return self.rollout_beam_search(state)

    def multi_rollout(
            self,
            state: Tensor,
            rollout_func,
            samples: int = 1,
    ) -> tuple[Action, float]:
        actions = [rollout_func(state)[0] for _ in range(samples)]

        action_counter = Counter(actions)

        action = action_counter.most_common(1)[0][0]

        return action, 0.0

    def rollout_beam_search(
            self,
            state: Tensor,
            beam_width: int = 32,
            depth: int = None,
            reward_decay: float = .95,
            use_done: bool = True,
            use_value_model: bool = True,
            value_model_weight: float = .2,
    ) -> tuple[Action, float]:
        @dataclass
        class Hypothesis:
            action: Tensor
            state: ValueModelInput
            reward: float
            not_done: float = 1.

            def clone(self) -> 'Hypothesis':
                return Hypothesis(
                    action=self.action.clone(),
                    state=ValueModelInput(
                        cont_state=self.state.cont_state.clone(),
                        disc_state=self.state.disc_state.clone(),
                    ),
                    reward=self.reward,
                    not_done=self.not_done,
                )

        device = state.device

        state = ValueModelInput.from_raw_state(state.unsqueeze(0))
        state.cont_state = state.cont_state.squeeze(0)
        state.disc_state = state.disc_state.squeeze(0)

        action_count = 2 ** 4

        action_options = torch.tensor([
            [-1, -1, -1, -1],
            [-1, -1, -1, +1],
            [-1, -1, +1, -1],
            [-1, -1, +1, +1],
            [-1, +1, -1, -1],
            [-1, +1, -1, +1],
            [-1, +1, +1, -1],
            [-1, +1, +1, +1],
            [+1, -1, -1, -1],
            [+1, -1, -1, +1],
            [+1, -1, +1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, -1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
            [+1, +1, +1, +1],
        ], device=device)

        sentinel_action = torch.empty(1)
        all_hypotheses = [Hypothesis(action=sentinel_action, state=state, reward=0.)]

        depth_i = -1
        for depth_i in range(depth) if depth is not None else count():
            new_hypotheses = []
            for h in all_hypotheses:
                for i in range(action_count):
                    new_h = h.clone()
                    if h.action is sentinel_action:
                        new_h.action = action_options[i]
                    new_hypotheses.append(new_h)
            all_hypotheses = new_hypotheses

            all_states = [h.state for h in all_hypotheses]
            all_states_cont = torch.stack([s.cont_state for s in all_states])
            all_states_disc = torch.stack([s.disc_state for s in all_states])
            value_model_input = ValueModelInput(cont_state=all_states_cont, disc_state=all_states_disc)

            actions = action_options.repeat(all_states_cont.size(0) // action_count, 1)

            world_model_input = WorldModelInput.from_value_model_input(value_model_input, actions)
            pred: WorldModelOutput = self.world_model.predict(world_model_input)

            pred_reward = pred.reward.squeeze(1)

            if use_value_model:
                pred_value = value_model_weight * self.value_model(value_model_input)
                pred_reward += pred_value
                self.stats.reward_value_ratio.append((pred_reward / pred_value).mean().item())

            pred_state_cont = pred.next_cont_state.split(1, dim=0)
            # pred_state_disc = pred.next_disc_state_thresholded.split(1, dim=0)
            pred_state_disc = pred.next_disc_state.split(1, dim=0)
            pred_reward = pred_reward.cpu().numpy()
            pred_not_done = (1 - pred.done_prob).cpu().numpy()

            for h, cont_state, disc_state, reward, not_done in zip_require_same_len(
                    all_hypotheses,
                    pred_state_cont,
                    pred_state_disc,
                    pred_reward,
                    pred_not_done,
            ):
                h.state = ValueModelInput(cont_state.squeeze(0), disc_state.squeeze(0))
                h.reward += reward * (reward_decay ** depth_i) * h.not_done

                if use_done:
                    h.not_done *= not_done

            if len(all_hypotheses) > beam_width:
                all_hypotheses = heapq.nlargest(beam_width, all_hypotheses, key=lambda h: h.reward)

            first_action = all_hypotheses[0].action
            only_one_action = all(h.action.equal(first_action) for h in all_hypotheses)
            if only_one_action:
                break

        self.stats and self.stats.lookahead_depth.append(depth_i + 1)

        best_hypothesis = max(all_hypotheses, key=lambda h: h.reward)
        best_action = best_hypothesis.action.cpu().numpy()

        return best_action, best_hypothesis.reward


class RandomAgentWrapper(Agent):
    def __init__(self, agent: WalkerAgent, random_steps: int = 10000):
        self.agent = agent
        self.random_steps = random_steps
        self.steps = 0

    @property
    def stats(self) -> Stats:
        return self.agent.stats

    def reset(self) -> None:
        self.agent.reset()

    def get_action(self, state: Tensor) -> Action:
        if self.steps < self.random_steps:
            random_prob = 1 - self.steps / self.random_steps
            self.stats.random_action_prob.append(random_prob)

            self.steps += 1
            if random.random() < random_prob:
                return self.agent._random_action()

        return self.agent.get_action(state)

    def eval(self) -> None:
        self.agent.eval()

    def train(self) -> None:
        self.agent.train()
