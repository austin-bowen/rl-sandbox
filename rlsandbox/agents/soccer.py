from abc import abstractmethod
from math import atan2, sqrt, pi
from random import Random
from typing import NamedTuple, Tuple

import numpy as np
import torch
from torch import nn

from rlsandbox.agents.agent import Agent
from rlsandbox.envs.soccer import SoccerState, SoccerAction, Agent as EnvAgent
from rlsandbox.math import trunc_angle
from rlsandbox.types import Location2D


class BaseSoccerAgent(Agent):
    @abstractmethod
    def get_action(self, state: SoccerState) -> SoccerAction:
        ...

    def _build_observation(self, state: SoccerState) -> 'Observation':
        agent = state.agent

        ball_angle, ball_dist = self._get_angle_and_dist(agent, state.ball.location)

        ball = state.ball
        ball_speed = sqrt(ball.velocity.dx ** 2 + ball.velocity.dy ** 2)
        ball_speed_angle = atan2(ball.velocity.dy, ball.velocity.dx)
        ball_speed_angle -= agent.heading

        left_goal_post_angle, left_goal_post_dist = self._get_angle_and_dist(agent, state.goal.left_post_location)
        right_goal_post_angle, right_goal_post_dist = self._get_angle_and_dist(agent, state.goal.right_post_location)

        return Observation(
            ball_angle=ball_angle / pi,
            ball_dist=sqrt(ball_dist),
            ball_speed=ball_speed,
            ball_speed_angle=ball_speed_angle / pi,
            left_goal_post_angle=left_goal_post_angle / pi,
            left_goal_post_dist=sqrt(left_goal_post_dist),
            right_goal_post_angle=right_goal_post_angle / pi,
            right_goal_post_dist=sqrt(right_goal_post_dist),
        )

    def _get_angle_and_dist(self, agent: EnvAgent, location: Location2D) -> Tuple[float, float]:
        dx = location.x - agent.location.x
        dy = location.y - agent.location.y

        angle = atan2(dy, dx) - agent.heading
        angle = trunc_angle(angle)
        dist = sqrt(dx ** 2 + dy ** 2)

        return angle, dist


class SimpleSoccerAgent(BaseSoccerAgent):
    rng: Random

    def __init__(self, rng: Random = None):
        self.rng = rng or Random()

    def get_action(self, state: SoccerState) -> SoccerAction:
        obs = self._build_observation(state)

        return SoccerAction(
            move_dist=self.rng.random(),
            turn_angle=0.2 * obs.ball_angle,
            kick_strength=self.rng.random(),
        )


class ANNSoccerAgent(BaseSoccerAgent):
    model: 'SoccerAgentModel'

    def __init__(self, model: 'SoccerAgentModel' = None):
        self.model = model or SoccerAgentModel()

    def get_action(self, state: SoccerState) -> SoccerAction:
        obs = self._build_observation(state)

        self.model.eval()

        obs = obs.to_tensor()
        obs = obs.unsqueeze(0)

        action: torch.Tensor = self.model(obs)
        action = action.detach().numpy()

        move_dist, turn_angle, kick_strength = action[0, :]

        return SoccerAction(
            move_dist=sigmoid(move_dist),
            turn_angle=pi * (sigmoid(turn_angle) - 0.5),
            # kick_strength=min(max(0., kick_strength), 1.),
            kick_strength=sigmoid(kick_strength),
        )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Observation(NamedTuple):
    ball_angle: float
    ball_dist: float
    ball_speed: float
    ball_speed_angle: float

    left_goal_post_angle: float
    left_goal_post_dist: float
    right_goal_post_angle: float
    right_goal_post_dist: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self, dtype=torch.float32)


class SoccerAgentModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(8, 8),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Sigmoid(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
