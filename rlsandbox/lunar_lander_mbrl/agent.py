import heapq
import random
from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from rlsandbox.base.utils import zip_require_same_len
from rlsandbox.lunar_lander_mbrl.logging import log_metrics
from rlsandbox.lunar_lander_mbrl.model import LunarLanderWorldModel
from rlsandbox.lunar_lander_mbrl.utils import assert_shape, printne


class Stats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.lookahead_depth = []
        self.reward_value_ratio = []

    def emit(self, step: int = None) -> None:
        metrics = dict()

        if self.lookahead_depth:
            metrics['agent_lookahead_depth'] = np.mean(self.lookahead_depth)

        if self.reward_value_ratio:
            metrics['agent_reward_value_ratio'] = np.mean(self.reward_value_ratio)

        if metrics:
            log_metrics(metrics, step=step)

        self.reset()


class LunarLanderAgent:
    def __init__(
            self,
            world_models: list[LunarLanderWorldModel],
            value_model,
    ):
        self.world_models = world_models
        self.value_model = value_model
        self.stats = Stats()

    @property
    def random_model(self) -> LunarLanderWorldModel:
        return random.choice(self.world_models)

    def get_action(
            self,
            state: Tensor,
    ) -> int:
        action, reward = self.rollout(state)

        return action

    def eval(self) -> None:
        for model in self.world_models:
            model.eval()

        try:
            self.value_model.eval()
        except AttributeError:
            pass

    def train(self) -> None:
        for model in self.world_models:
            model.train()

        try:
            self.value_model.train()
        except AttributeError:
            pass

    def rollout(self, state: Tensor) -> tuple[int, float]:
        if 1:
            try:
                self.__count += 1
            except AttributeError:
                self.__count = 1
            count = self.__count

            if random.random() > count / 10000:
                return np.random.randint(0, 4), 0.0

        return self.multi_rollout(
            state,
            # rollout_func=self.rollout0,
            rollout_func=self.rollout1,
            # rollout_func=self.rollout_beam_search,
        )

    def multi_rollout(
            self,
            state: Tensor,
            rollout_func,
            samples: int = 9,
    ) -> tuple[int, float]:
        actions = [rollout_func(state)[0] for _ in range(samples)]

        action_counter = Counter(actions)

        action = action_counter.most_common(1)[0][0]

        # action_counter = [action_counter[i] for i in range(4)]
        # action_counter = torch.tensor(action_counter, dtype=torch.float16)
        # action = F.softmax(action_counter, dim=0)
        # action = torch.multinomial(action, 1).item()

        return action, 0.0

    def rollout0(
            self,
            state: Tensor,
            action_repeats: int = 2,
            depth: int = 5,
            reward_decay: float = 0.9,
            explore: bool = True,
    ) -> tuple[int, float]:
        device = state.device
        action_count = 4
        action_options = torch.tensor(range(action_count), device=device)
        reward_stack = []
        done_stack = []
        pred_state = state.unsqueeze(0)
        too_uncertain = False

        for depth_i in range(depth):
            batch_size = action_count ** (depth_i + 1)

            pred_state = pred_state.repeat_interleave(action_count, dim=0)
            assert_shape(pred_state, (batch_size, state.shape[0]))

            action = action_options.repeat(pred_state.shape[0] // action_count)
            assert_shape(action, (batch_size,))

            pred_reward = torch.zeros(batch_size, device=device)
            pred_done = torch.zeros(batch_size, device=device)

            # action_repeats = min(depth_i + 1, 4)
            for _ in range(action_repeats if depth_i else 1):
                all_model_preds = [
                    model(
                        state=pred_state,
                        action=action,
                    ) for model in self.world_models
                ]
                all_model_preds = [torch.cat(it, dim=1) for it in all_model_preds]

                pred_state_diff_and_reward = []
                for all_preds in all_model_preds:
                    assert_shape(all_preds, (batch_size, 10))

                    # Convert leg logits to probabilities
                    all_preds[:, 6:8] = F.sigmoid(all_preds[:, 6:8])

                    all_preds = all_preds.clone()
                    all_preds[:, :6] -= pred_state[:, :6]
                    all_preds[:, 6:8] -= 0.5
                    all_preds[:, 9] -= 0.5

                    pred_state_diff_and_reward.append(all_preds)

                # TODO remove?
                # norm = torch.maximum(
                #     torch.abs(pred_state_diff_and_reward[0]),
                #     torch.abs(pred_state_diff_and_reward[1]),
                # )
                # pred_state_diff_and_reward[0] /= norm
                # pred_state_diff_and_reward[1] /= norm

                eps = 0
                agreement = F.cosine_similarity(
                    pred_state_diff_and_reward[0] + eps,
                    pred_state_diff_and_reward[1] + eps,
                    dim=1,
                )

                if explore:
                    explore = False

                    should_explore = random.random() > agreement.min().item()
                    # should_explore = random.random() > agreement.mean().item()
                    if should_explore:
                        printne('?')
                        # action = agreement.argmin().item()
                        action = np.random.choice(
                            [0, 1, 2, 3],
                            p=F.softmax(-agreement, dim=0).cpu().numpy(),
                        )
                        return action, 0.

                mean_agreement = agreement.mean().item()
                if random.random() > mean_agreement:
                    too_uncertain = True
                    break

                pred_state_and_reward = (all_model_preds[0] + all_model_preds[1]) / 2

                pred_state = pred_state_and_reward[:, :8]
                pred_reward += pred_state_and_reward[:, 8] * (agreement + 1) / 2

                # TODO accumulate?
                pred_done_logit = pred_state_and_reward[:, 9]
                pred_done = F.sigmoid(pred_done_logit)
                # pred_done = torch.maximum(pred_done, pred_done_)

            is_last = depth_i + 1 == depth
            if 0 and is_last and self.value_model.has_items():
                pred_value = self.value_model.predict(pred_state.detach().cpu().numpy())
                pred_value = torch.from_numpy(pred_value).to(device)
                print('pred_value:')
                print(pred_value)
                pred_reward += pred_value * 1.
            if 0 and is_last:
                pred_value = self.value_model(pred_state)
                print('pred_value:', pred_value)
                pred_reward += pred_value

            reward_stack.append(pred_reward)
            done_stack.append(pred_done)

            if too_uncertain:
                break

        printne(depth_i)

        reward_stack.reverse()
        done_stack.reverse()

        for reward, prev_reward, prev_done in zip(reward_stack, reward_stack[1:], done_stack[1:]):
            assert_shape(reward, (prev_reward.shape[0] * action_count,))

            for i in range(prev_reward.shape[0]):
                start = i * action_count
                agg_reward = reward[start:start + action_count].max()
                done_weight = 1 - prev_done[i]
                # done_weight = 1 - (prev_done[i] >= 0.1).float()
                prev_reward[i] += agg_reward * reward_decay * done_weight

        reward = reward_stack[-1]
        assert_shape(reward, (action_count,))

        action = reward.argmax().item()

        reward = reward[action].item()

        return action, reward

    def rollout1(
            self,
            state: Tensor,
            depth: int = 10,
            leg_threshold: float = None,
            reward_decay: float = .9,
            use_value_model: bool = True,
            value_model_weight: float = .1,
    ) -> tuple[int, float]:
        model_selector = lambda: self.random_model

        model = model_selector()
        device = state.device
        actions = torch.tensor([0, 1, 2, 3], device=device)
        actions_4x = actions.repeat(4)

        pred_state = state.unsqueeze(0)
        pred_state = pred_state.repeat_interleave(4, dim=0)
        assert_shape(pred_state, (4, state.shape[0]))

        pred_state, pred_reward, pred_done_logit = model(pred_state, actions)
        pred_state[:, 6:8] = F.sigmoid(pred_state[:, 6:8])
        if leg_threshold is not None:
            pred_state[:, 6:8] = (pred_state[:, 6:8] > leg_threshold).float()
        pred_reward = pred_reward.squeeze(1)
        pred_done = F.sigmoid(pred_done_logit.squeeze(1))

        if use_value_model:
            pred_reward += value_model_weight * self.value_model(pred_state)

        reward_stack = [pred_reward]
        done_stack = [pred_done]

        for depth_i in range(1, depth):
            pred_state = pred_state.repeat_interleave(4, dim=0)
            assert_shape(pred_state, (4 * 4, state.shape[0]))

            model = model_selector()
            pred_state, pred_reward, pred_done_logit = model(pred_state, actions_4x)
            pred_state[:, 6:8] = F.sigmoid(pred_state[:, 6:8])
            if leg_threshold is not None:
                pred_state[:, 6:8] = (pred_state[:, 6:8] > leg_threshold).float()
            pred_reward = pred_reward.squeeze(1)
            pred_done = F.sigmoid(pred_done_logit.squeeze(1))

            if use_value_model:
                pred_reward += value_model_weight * self.value_model(pred_state)

            pred_reward = pred_reward.view(4, 4)

            best_actions = pred_reward.argmax(dim=1)
            # best_actions = torch.tensor([
            #     np.random.randint(0, 4),
            #     np.random.randint(0, 4),
            #     np.random.randint(0, 4),
            #     np.random.randint(0, 4),
            # ], device=device)
            # best_actions = F.softmax(pred_reward, dim=1)
            # best_actions = torch.multinomial(best_actions, num_samples=1).squeeze(1)
            assert_shape(best_actions, (4,))

            pred_state = pred_state.view(4, 4, -1)
            pred_state = pred_state[range(4), best_actions]
            assert_shape(pred_state, (4, state.shape[0]))

            reward_alpha = 1.
            pred_reward = (
                    reward_alpha * pred_reward.max(dim=1).values
                    + (1 - reward_alpha) * pred_reward.min(dim=1).values
            )
            pred_reward *= reward_decay ** depth_i
            assert_shape(pred_reward, (4,))

            pred_done = pred_done.view(4, 4)
            pred_done = pred_done[range(4), best_actions]

            reward_stack.append(pred_reward)
            done_stack.append(pred_done)

        use_done = 1
        if use_done:
            rewards = torch.zeros((4,), device=device)

            rewards_backwards = reward_stack[::-1]
            dones_backwards = done_stack[::-1]

            for i in range(depth - 1):
                rewards = (rewards_backwards[i] + rewards) * (1 - dones_backwards[i + 1])

            rewards = rewards + rewards_backwards[-1]
        else:
            rewards = torch.stack(reward_stack)
            assert_shape(rewards, (depth, 4))

            rewards = torch.sum(rewards, dim=0)
        assert_shape(rewards, (4,))

        action = rewards.argmax().item()

        reward = rewards[action].item()

        return action, reward

    def rollout_beam_search(
            self,
            state: Tensor,
            beam_width: int = 32,
            depth: int = None,
            leg_threshold: float = None,
            reward_decay: float = .9,
            use_done: bool = False,
            use_value_model: bool = True,
            value_model_weight: float = .1,
    ) -> tuple[int, float]:
        @dataclass
        class Hypothesis:
            action: int
            state: Tensor
            reward: float
            not_done: float = 1.

            def clone(self) -> 'Hypothesis':
                return Hypothesis(
                    action=self.action,
                    state=self.state.clone(),
                    reward=self.reward,
                    not_done=self.not_done,
                )

        device = state.device

        action_count = 4
        action_options = torch.tensor(range(action_count), device=device)

        model_selector = lambda: self.random_model

        all_hypotheses = [Hypothesis(action=None, state=state, reward=0.)]

        depth_i = -1
        for depth_i in range(depth) if depth is not None else count():
            new_hypotheses = []
            for h in all_hypotheses:
                for i in range(action_count):
                    new_h = h.clone()
                    if h.action is None:
                        new_h.action = i
                    new_hypotheses.append(new_h)
            all_hypotheses = new_hypotheses

            all_states = [h.state for h in all_hypotheses]
            all_states = torch.stack(all_states)

            actions = action_options.repeat(all_states.size(0) // action_count)

            pred_state, pred_reward, pred_done = model_selector()(all_states, actions)
            pred_state[:, 6:8] = F.sigmoid(pred_state[:, 6:8])
            if leg_threshold is not None:
                pred_state[:, 6:8] = (pred_state[:, 6:8] > leg_threshold).float()
            pred_reward = pred_reward.squeeze(1)
            pred_done = F.sigmoid(pred_done.squeeze(1))

            if use_value_model:
                pred_value = value_model_weight * self.value_model(pred_state)
                pred_reward += pred_value
                self.stats.reward_value_ratio.append((pred_reward / pred_value).mean().item())

            pred_state = pred_state.split(1, dim=0)
            pred_reward = pred_reward.cpu().numpy()
            pred_not_done = (1 - pred_done).cpu().numpy()

            for h, state, reward, not_done in zip_require_same_len(
                    all_hypotheses,
                    pred_state,
                    pred_reward,
                    pred_not_done,
            ):
                h.state = state.squeeze(0)
                h.reward += reward * (reward_decay ** depth_i) * h.not_done

                if use_done:
                    h.not_done *= not_done

            if len(all_hypotheses) > beam_width:
                all_hypotheses = heapq.nlargest(beam_width, all_hypotheses, key=lambda h: h.reward)

            first_action = all_hypotheses[0].action
            only_one_action = all(h.action == first_action for h in all_hypotheses)
            if only_one_action:
                break

        self.stats and self.stats.lookahead_depth.append(depth_i + 1)

        best_hypothesis = max(all_hypotheses, key=lambda h: h.reward)

        return best_hypothesis.action, best_hypothesis.reward
