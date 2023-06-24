import os
import random
from abc import abstractmethod
from copy import deepcopy
from itertools import count
from typing import Tuple

import torch
from torch import nn
from torch.multiprocessing import get_context

from rlsandbox.base import Agent, Reward
from rlsandbox.env_runner import EnvRunner
from rlsandbox.monitor import Monitor
from rlsandbox.team_soccer.agent import ANNTeamSoccerAgent, BaseTeamSoccerAgent
from rlsandbox.team_soccer.env import TeamSoccerEnv, AgentId, TeamSoccerState, SoccerActions, TeamId
from rlsandbox.team_soccer.renderer import TeamSoccerEnvRenderer
from rlsandbox.types import Size2D


class AgentMux:
    @abstractmethod
    def __getitem__(self, item: AgentId) -> BaseTeamSoccerAgent:
        ...

    @abstractmethod
    def agents(self) -> list[BaseTeamSoccerAgent]:
        ...


class SingleAgentMux(AgentMux):
    def __init__(self, agent: BaseTeamSoccerAgent):
        self.agent = agent

    def __getitem__(self, item: AgentId) -> BaseTeamSoccerAgent:
        return self.agent

    def agents(self) -> list[BaseTeamSoccerAgent]:
        return [self.agent]


class OneAgentPerTeamMux(AgentMux):
    def __init__(self, left_agent: BaseTeamSoccerAgent, right_agent: BaseTeamSoccerAgent):
        self.left_agent = left_agent
        self.right_agent = right_agent

    def __getitem__(self, item: AgentId) -> BaseTeamSoccerAgent:
        return self.left_agent if item.team == TeamId.LEFT else self.right_agent

    def agents(self) -> list[BaseTeamSoccerAgent]:
        return [self.left_agent, self.right_agent]


class MultiAgent(Agent):
    agent_mux: AgentMux

    def __init__(self, agent_mux: AgentMux):
        self.agent_mux = agent_mux

    def reset(self) -> None:
        for agent in self.agent_mux.agents():
            agent.reset()

    def get_action(self, state: TeamSoccerState) -> SoccerActions:
        return {
            agent_id: self.agent_mux[agent_id].get_action(state, agent_id)
            for agent_id in state.agent_ids
        }


def main(pool):
    field_size = Size2D(40, 20)
    env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=2,
        right_team_size=2,
        max_steps=200,
        max_steps_no_ball_movement=50,
    )

    monitor_env = TeamSoccerEnv(
        field_size=Size2D(40, 20),
        left_team_size=env.left_team_size,
        right_team_size=env.right_team_size,
        # max_steps_no_ball_movement=50,
    )

    env_renderer = TeamSoccerEnvRenderer(monitor_env, fps=20, scale=20)

    agent = ANNTeamSoccerAgent(obs_dim=24)

    best_agents = [agent]
    opponent = deepcopy(agent)
    opponent_update_delay = 50

    agent_mux = OneAgentPerTeamMux(left_agent=opponent, right_agent=agent)
    multi_agent = MultiAgent(agent_mux)

    games_per_eval = 64

    new_bests_found = 0

    with Monitor(monitor_env, env_renderer, update_agent_on='step') as monitor:
        monitor.set_agent(multi_agent)

        for gens in count():
            print(f'Generation {gens}')

            opponent_index = max(0, new_bests_found - opponent_update_delay)
            opponent = deepcopy(best_agents[opponent_index])
            agent_mux.left_agent = opponent
            monitor.set_agent(multi_agent)

            # if gens % opponent_update_period == 0:
            #     opponent = deepcopy(agent)
            #     agent_mux.left_agent = opponent
            #     monitor.set_agent(multi_agent)

            new_agent = mutate_agent(agent)

            scores = pool.starmap(evaluate_agent, ((env, opponent, agent, new_agent) for _ in range(games_per_eval)))
            scores.reverse()
            reward_decay = 0.99
            agent_reward = sum(reward_decay ** i * score[0] for i, score in enumerate(scores)) / games_per_eval
            new_agent_reward = sum(reward_decay ** i * score[1] for i, score in enumerate(scores)) / games_per_eval
            # agent_reward = tuple(agent_reward)
            # new_agent_reward = tuple(new_agent_reward)

            # new_agent_wins = sum(
            #     tuple(score[1]) >= tuple(score[0]) for score in scores
            # )
            # agent_wins = games_per_eval - new_agent_wins

            print(
                f'Agent reward: {agent_reward};\t'
                f'new agent reward: {new_agent_reward};\t'
                # f'agent wins: {agent_wins};\t'
                # f'new agent wins: {new_agent_wins};\t'
                f'new bests found: {new_bests_found};\t'
            )

            if new_agent_reward >= agent_reward:
                # if new_agent_wins > agent_wins:
                # if new_agent_reward >= agent_reward and new_agent_wins >= agent_wins:
                new_bests_found += 1
                best_agents.append(new_agent)

                agent = new_agent

                agent_mux.right_agent = agent
                monitor.set_agent(multi_agent)


def main_direct_compare(pool):
    field_size = Size2D(40, 20)
    env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=2,
        right_team_size=2,
        max_steps=200,
        max_steps_no_ball_movement=50,
    )

    monitor_env = TeamSoccerEnv(
        field_size=Size2D(40, 20),
        left_team_size=env.left_team_size,
        right_team_size=env.right_team_size,
        # max_steps_no_ball_movement=50,
    )

    env_renderer = TeamSoccerEnvRenderer(monitor_env, fps=20, scale=20)

    agent = ANNTeamSoccerAgent(obs_dim=24)

    agent_mux = OneAgentPerTeamMux(left_agent=agent, right_agent=agent)
    multi_agent = MultiAgent(agent_mux)

    games_per_eval = 1

    new_bests_found = 0

    with Monitor(monitor_env, env_renderer, update_agent_on='step') as monitor:
        monitor.set_agent(multi_agent)

        for gens in count():
            print(f'Generation {gens}')

            new_agent = mutate_agent(agent)

            agent_mux.right_agent = new_agent
            monitor.set_agent(multi_agent)

            scores = pool.starmap(evaluate_agent2, ((env, agent, new_agent) for _ in range(games_per_eval)))
            scores.reverse()
            reward_decay = 1.
            agent_reward = sum(reward_decay ** i * score[0] for i, score in enumerate(scores)) / games_per_eval
            new_agent_reward = sum(reward_decay ** i * score[1] for i, score in enumerate(scores)) / games_per_eval
            # agent_reward = tuple(agent_reward)
            # new_agent_reward = tuple(new_agent_reward)

            new_agent_wins = sum(
                # tuple(score[1]) > tuple(score[0]) for score in scores
                new_agent_score > best_agent_score
                for best_agent_score, new_agent_score in scores
            )
            agent_wins = games_per_eval - new_agent_wins

            print(
                f'Agent reward: {agent_reward};\t'
                f'new agent reward: {new_agent_reward};\t'
                f'new agent wins: {100 * new_agent_wins / games_per_eval:.2f}%;\t'
                f'new bests found: {new_bests_found};\t'
            )

            # if new_agent_reward >= agent_reward:
            if new_agent_wins > agent_wins:
                # if new_agent_reward >= agent_reward and new_agent_wins >= agent_wins:
                new_bests_found += 1

                agent = new_agent

                agent_mux.left_agent = agent
                monitor.set_agent(multi_agent)


def mutate_agent(agent: ANNTeamSoccerAgent):
    new_agent = ANNTeamSoccerAgent(obs_dim=agent.model.obs_dim)

    weight_change = 0.08
    max_weight = 10
    weight_decay = max_weight / (max_weight + weight_change)
    dropout = 0.

    for layer, new_layer in zip(agent.model.layers, new_agent.model.layers):
        if not isinstance(layer, nn.Linear):
            continue

        change = weight_change * (2 * torch.rand(layer.weight.shape) - 1)
        change = nn.functional.dropout(change, p=dropout)
        new_layer.weight = nn.Parameter((layer.weight + change) * weight_decay)

        if layer.bias is not None:
            change = weight_change * (2 * torch.rand(layer.bias.shape) - 1)
            change = nn.functional.dropout(change, p=dropout)
            new_layer.bias = nn.Parameter((layer.bias + change) * weight_decay)

    return new_agent


def evaluate_agent(env, opponent, best_agent, new_agent) -> Tuple[Reward, Reward]:
    env_seed = hash((
        os.getpid(),
        random.randint(0, 2 ** 32 - 1),
    ))

    # Best agent vs opponent
    env.rng.seed(env_seed)
    agent_mux = OneAgentPerTeamMux(left_agent=opponent, right_agent=best_agent)
    multi_agent = MultiAgent(agent_mux)
    runner = EnvRunner(env, multi_agent)

    state_changes = runner.run()
    all_rewards = [state_change.reward for state_change in state_changes]

    best_agent_rewards = []
    for agent_rewards in all_rewards:
        for agent_id, reward in agent_rewards.items():
            if agent_id.team == TeamId.RIGHT:
                best_agent_rewards.append(reward)

    # New agent vs opponent
    env.rng.seed(env_seed)
    agent_mux = OneAgentPerTeamMux(left_agent=opponent, right_agent=new_agent)
    multi_agent = MultiAgent(agent_mux)
    runner = EnvRunner(env, multi_agent)

    state_changes = runner.run()
    all_rewards = [state_change.reward for state_change in state_changes]

    new_agent_rewards = []
    for agent_rewards in all_rewards:
        for agent_id, reward in agent_rewards.items():
            if agent_id.team == TeamId.RIGHT:
                new_agent_rewards.append(reward)

    total_best_agent_rewards = sum(best_agent_rewards)
    total_new_agent_rewards = sum(new_agent_rewards)

    return total_best_agent_rewards, total_new_agent_rewards


def evaluate_agent2(
        env: TeamSoccerEnv,
        best_agent: BaseTeamSoccerAgent,
        new_agent: BaseTeamSoccerAgent,
) -> Tuple[Reward, Reward]:
    env_seed = hash((
        os.getpid(),
        random.randint(0, 2 ** 32 - 1),
    ))

    best_agent_reward_total = 0
    new_agent_reward_total = 0

    for best_agent_team, new_agent_team in (
            (TeamId.LEFT, TeamId.RIGHT),
            (TeamId.RIGHT, TeamId.LEFT),
    ):
        env.rng.seed(env_seed)

        multi_agent = MultiAgent(
            OneAgentPerTeamMux(
                left_agent=best_agent if best_agent_team == TeamId.LEFT else new_agent,
                right_agent=new_agent if new_agent_team == TeamId.RIGHT else best_agent,
            ),
        )

        env_runner = EnvRunner(env, multi_agent)

        state_changes = env_runner.run()

        all_rewards = [it.reward for it in state_changes]

        for agent_rewards in all_rewards:
            for agent_id, reward in agent_rewards.items():
                if agent_id.team == best_agent_team:
                    best_agent_reward_total += reward
                else:
                    new_agent_reward_total += reward

    return best_agent_reward_total, new_agent_reward_total


if __name__ == '__main__':
    with get_context('forkserver').Pool() as pool:
        main_direct_compare(pool)
