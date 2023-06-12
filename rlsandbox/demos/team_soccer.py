import random
from abc import abstractmethod
from itertools import count

import numpy as np
import torch
from torch import nn

from rlsandbox.agents.agent import Agent
from rlsandbox.agents.team_soccer import ANNTeamSoccerAgent, SimpleTeamSoccerAgent, BaseTeamSoccerAgent
from rlsandbox.envs.renderers.team_soccer_renderer import TeamSoccerEnvRenderer
from rlsandbox.envs.team_soccer import TeamSoccerEnv, AgentId, TeamSoccerState, SoccerActions
from rlsandbox.monitor import Monitor
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
        left_team_size=1,
        right_team_size=1,
        max_steps=300,
    )

    env_renderer = TeamSoccerEnvRenderer(env, fps=30, scale=30)

    agent = SimpleTeamSoccerAgent()
    agent_mux = SingleAgentMux(agent)
    uber_agent = UberAgent(agent_mux)

    with Monitor(env, env_renderer) as monitor:
        monitor.set_agent(uber_agent)
        input('Press Enter to exit')


def main(pool):
    field_size = Size2D(40, 20)
    env = TeamSoccerEnv(
        field_size=field_size,
        left_team_size=0,
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

    agent = ANNTeamSoccerAgent()
    agent.reset()

    games_per_eval = 64

    new_bests_found = 0

    with Monitor(monitor_env, env_renderer) as monitor:
        monitor.set_agent(agent)

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
                monitor.set_agent(agent)


def mutate_agent(agent: ANNTeamSoccerAgent):
    new_agent = ANNTeamSoccerAgent()

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


def evaluate_agent(env, best_agent, new_agent):
    env_seed = random.randint(0, 2 ** 32 - 1)

    scores = []

    for agent in [best_agent, new_agent]:
        env.rng.seed(env_seed)
        state = env.reset()
        total_reward = np.zeros(2)

        agent.reset()

        while True:
            action = agent.get_action(state)

            state_change = env.step(action)

            total_reward += state_change.reward

            if state_change.done:
                break

        scores.append(total_reward)

    return scores


if __name__ == '__main__':
    main_simple()
    # with get_context('fork').Pool() as pool:
    #     main(pool)
