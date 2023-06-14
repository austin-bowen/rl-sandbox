from abc import abstractmethod
from itertools import count
from multiprocessing import get_context
from typing import Tuple

import torch
from torch import nn

from rlsandbox.agents.agent import Agent
from rlsandbox.agents.team_soccer import ANNTeamSoccerAgent, SimpleTeamSoccerAgent, BaseTeamSoccerAgent
from rlsandbox.env_runner import EnvRunner
from rlsandbox.envs.renderers.team_soccer_renderer import TeamSoccerEnvRenderer
from rlsandbox.envs.team_soccer import TeamSoccerEnv, AgentId, TeamSoccerState, SoccerActions, TeamId
from rlsandbox.monitor import Monitor
from rlsandbox.types import Size2D, Reward


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


class UberAgent(Agent):
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


def main_simple():
    field_size = Size2D(40, 20)
    env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=3,
        right_team_size=3,
        max_steps=300,
    )

    env_renderer = TeamSoccerEnvRenderer(env, fps=30, scale=30)

    agent = SimpleTeamSoccerAgent()
    # agent_mux = SingleAgentMux(agent)
    agent_mux = OneAgentPerTeamMux(agent, agent)
    uber_agent = UberAgent(agent_mux)

    with Monitor(env, env_renderer) as monitor:
        monitor.set_agent(uber_agent)
        input('Press Enter to exit')


def main(pool):
    field_size = Size2D(40, 20)
    env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=1,
        right_team_size=1,
        max_steps=100,
    )

    monitor_env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=env.left_team_size,
        right_team_size=env.right_team_size,
        max_steps=300,
    )

    env_renderer = TeamSoccerEnvRenderer(monitor_env, fps=30, scale=30)

    agent = ANNTeamSoccerAgent(obs_dim=21)
    agent_mux = OneAgentPerTeamMux(agent, agent)
    uber_agent = UberAgent(agent_mux)

    games_per_eval = 64

    new_bests_found = 0

    with Monitor(monitor_env, env_renderer) as monitor:
        monitor.set_agent(uber_agent)

        for gens in count():
            print(f'Generation {gens}')

            new_agent = mutate_agent(agent)

            scores = pool.starmap(evaluate_agent, ((env, agent, new_agent) for _ in range(games_per_eval)))
            agent_reward = sum(score[0] for score in scores) / games_per_eval
            new_agent_reward = sum(score[1] for score in scores) / games_per_eval
            agent_reward = tuple(agent_reward)
            new_agent_reward = tuple(new_agent_reward)

            new_agent_wins = sum(
                tuple(score[1]) >= tuple(score[0]) for score in scores
            )
            agent_wins = games_per_eval - new_agent_wins

            print(
                f'Agent reward: {agent_reward};\t'
                f'new agent reward: {new_agent_reward};\t'
                f'agent wins: {agent_wins};\t'
                f'new agent wins: {new_agent_wins};\t'
                f'new bests found: {new_bests_found};\t'
            )

            if new_agent_reward >= agent_reward:
                # if new_agent_wins > agent_wins:
                # if new_agent_reward >= agent_reward and new_agent_wins >= agent_wins:
                new_bests_found += 1

                agent = new_agent

                agent_mux.left_agent = agent_mux.right_agent
                agent_mux.right_agent = agent
                monitor.set_agent(uber_agent)


def mutate_agent(agent: ANNTeamSoccerAgent):
    new_agent = ANNTeamSoccerAgent(obs_dim=agent.model.obs_dim)

    weight_change = 0.03
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


def evaluate_agent(env, best_agent, new_agent) -> Tuple[Reward, Reward]:
    env.rng.seed()

    agent_mux = OneAgentPerTeamMux(left_agent=best_agent, right_agent=new_agent)
    uber_agent = UberAgent(agent_mux)
    runner = EnvRunner(env, uber_agent)

    state_changes = runner.run()
    all_rewards = [state_change.reward for state_change in state_changes]

    left_rewards = []
    right_rewards = []
    for agent_rewards in all_rewards:
        for agent_id, reward in agent_rewards.items():
            if agent_id.team == TeamId.LEFT:
                left_rewards.append(reward)
            else:
                right_rewards.append(reward)

    total_left_rewards = sum(left_rewards)
    total_right_rewards = sum(right_rewards)

    return total_left_rewards, total_right_rewards


if __name__ == '__main__':
    # main_simple()
    with get_context('fork').Pool() as pool:
        main(pool)
