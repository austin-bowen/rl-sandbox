import random
from itertools import count
from multiprocessing import get_context

import torch
from torch import nn

from rlsandbox.agents.soccer import ANNSoccerAgent
from rlsandbox.envs.renderers.soccer_renderer import SoccerEnvRenderer
from rlsandbox.envs.soccer import SoccerEnv
from rlsandbox.monitor import Monitor
from rlsandbox.types import Size2D


def main(pool):
    field_size = Size2D(40, 20)
    env = SoccerEnv(field_size=field_size, max_steps=100)
    monitor_env = SoccerEnv(field_size=field_size, max_steps=300)
    # monitor_env = env

    env_renderer = SoccerEnvRenderer(monitor_env, fps=30, scale=30)

    agent = ANNSoccerAgent()
    agent.reset()

    games_per_eval = 50

    new_bests_found = 0

    with Monitor(monitor_env, env_renderer) as monitor:
        monitor.set_agent(agent)

        for gens in count():
            print(f'Generation {gens}')

            new_agent = mutate_agent(agent)

            scores = pool.starmap(evaluate_agent, ((env, agent, new_agent) for _ in range(games_per_eval)))
            agent_reward = sum(score[0] for score in scores) / games_per_eval
            new_agent_reward = sum(score[1] for score in scores) / games_per_eval
            # reward = sum(rewards) / games_per_eval
            # reward = sum(evaluate_agent(env, new_agent) for _ in range(games_per_eval)) / games_per_eval

            # print(f'Average reward: {reward:8.3f}; best reward: {best_reward:8.3f}')
            print(f'Agent reward: {agent_reward:8.3f}; '
                  f'new agent reward: {new_agent_reward:8.3f}; '
                  f'new bests found: {new_bests_found}')

            if new_agent_reward >= agent_reward:
                new_bests_found += 1

                # best_reward = reward
                agent = new_agent
                monitor.set_agent(agent)


def mutate_agent(agent: ANNSoccerAgent):
    new_agent = ANNSoccerAgent()

    weight_change = 0.05
    weight_decay = 10 / (10 + weight_change)
    dropout = 0.

    for layer, new_layer in zip(agent.model.layers, new_agent.model.layers):
        if not isinstance(layer, nn.Linear):
            continue

        change = weight_change * (2 * torch.rand(layer.weight.shape) - 1)
        change = nn.functional.dropout(change, p=dropout)
        new_layer.weight = nn.Parameter((layer.weight + change) * weight_decay)

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
        total_reward = 0.

        agent.reset()

        while True:
            action = agent.get_action(state)

            state_change = env.step(action)

            total_reward += state_change.reward

            if state_change.done:
                break

        scores.append(total_reward)

    return scores


def display_game(env, agent, env_renderer):
    state = env.reset()
    env_renderer.render(state)

    agent.reset()

    while True:
        action = agent.get_action(state)

        state_change = env.step(action)

        env_renderer.render(state_change.next_state)

        if state_change.done:
            break


if __name__ == '__main__':
    with get_context('fork').Pool(maxtasksperchild=100) as pool:
        main(pool)
