from abc import abstractmethod
from math import atan2, sqrt, pi, cos, sin
from random import Random
from typing import NamedTuple, List

import numpy as np
import torch
from torch import nn

from rlsandbox.agents.agent import Agent
from rlsandbox.envs.soccer import SoccerAction
from rlsandbox.envs.team_soccer import AgentId, TeamSoccerState, TeamSoccerAgent as EnvTeamSoccerAgent, TeamId
from rlsandbox.math import trunc_angle, unit_tanh
from rlsandbox.types import Location2D


class Observation(NamedTuple):
    ball_position: np.array
    ball_speed: float
    ball_speed_angle: float
    team_left_goal_post_position: np.array
    team_right_goal_post_position: np.array
    opponent_left_goal_post_position: np.array
    opponent_right_goal_post_position: np.array
    other_agent_positions: List[np.array]

    def to_tensor(self) -> torch.Tensor:
        array = np.concatenate([
            self.ball_position,
            [self.ball_speed, self.ball_speed_angle],
            self.team_left_goal_post_position,
            self.team_right_goal_post_position,
            self.opponent_left_goal_post_position,
            self.opponent_right_goal_post_position,
            *self.other_agent_positions,
        ])

        return torch.tensor(array, dtype=torch.float32)


class BaseTeamSoccerAgent(Agent):
    rng: Random

    def __init__(self, rng: Random = None):
        self.rng = rng or Random()

    @abstractmethod
    def get_action(self, state: TeamSoccerState, agent_id: AgentId) -> SoccerAction:
        ...

    def _build_observation(self, state: TeamSoccerState, agent_id: AgentId) -> 'Observation':
        agent = state.agent_with_id(agent_id)

        ball = state.ball
        ball_position = self._get_angle_and_dist(agent, ball.location)
        ball_speed = sqrt(ball.velocity.dx ** 2 + ball.velocity.dy ** 2)
        ball_speed_angle = atan2(ball.velocity.dy, ball.velocity.dx)
        ball_speed_angle -= agent.heading

        if agent.id.team == TeamId.LEFT:
            team_goal = state.left_goal
            opponent_goal = state.right_goal
        else:
            team_goal = state.right_goal
            opponent_goal = state.left_goal

        other_agents = [it for it in state.agents if it.id != agent_id]
        other_agent_positions = []
        for other_agent in other_agents:
            is_opponent = 1. if other_agent.id.team != agent.id.team else 0.
            position = self._get_angle_and_dist(agent, other_agent.location)

            other_agent_positions.append(
                np.concatenate(([is_opponent], position))
            )

        self.rng.shuffle(other_agent_positions)

        return Observation(
            ball_position=ball_position,
            ball_speed=ball_speed,
            ball_speed_angle=ball_speed_angle / pi,
            team_left_goal_post_position=self._get_angle_and_dist(agent, team_goal.left_post_location),
            team_right_goal_post_position=self._get_angle_and_dist(agent, team_goal.right_post_location),
            opponent_left_goal_post_position=self._get_angle_and_dist(agent, opponent_goal.left_post_location),
            opponent_right_goal_post_position=self._get_angle_and_dist(agent, opponent_goal.right_post_location),
            other_agent_positions=other_agent_positions,
        )

    def _get_angle_and_dist(self, agent: EnvTeamSoccerAgent, location: Location2D) -> np.array:
        dx = location.x - agent.location.x
        dy = location.y - agent.location.y

        angle = atan2(dy, dx) - agent.heading
        angle = trunc_angle(angle)

        dist = sqrt(dx ** 2 + dy ** 2)

        return np.array((
            cos(angle),
            sin(angle),
            self._dist_to_closeness(dist),
        ))

    @staticmethod
    def _dist_to_closeness(dist: float) -> float:
        assert dist >= 0
        return 1 / (1 + 0.5 * dist)


class SimpleTeamSoccerAgent(BaseTeamSoccerAgent):
    def get_action(self, state: TeamSoccerState, agent_id: AgentId) -> SoccerAction:
        obs = self._build_observation(state, agent_id)

        ball_angle_left = obs.ball_position[1]

        return SoccerAction(
            move_dist=self.rng.random(),
            turn_angle=0.2 * ball_angle_left,
            kick_strength=self.rng.random(),
        )


class ANNTeamSoccerAgent(BaseTeamSoccerAgent):
    model: 'SoccerAgentModel'

    def __init__(self, obs_dim: int = -1, model: 'SoccerAgentModel' = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or SoccerAgentModel(obs_dim)

    def get_action(self, state: TeamSoccerState, agent_id: AgentId) -> SoccerAction:
        obs = self._build_observation(state, agent_id)

        obs = obs.to_tensor()
        obs = obs.unsqueeze(0)

        self.model.eval()
        action: torch.Tensor = self.model(obs)
        action = action.detach().numpy()

        move_dist, turn_angle, kick_strength = action[0, :]

        return SoccerAction(
            move_dist=unit_tanh(move_dist),
            # move_dist=min(max(0, move_dist), 1),
            turn_angle=pi / 4 * np.tanh(turn_angle),
            # turn_angle=pi / 4 * min(max(-1, turn_angle), 1),
            kick_strength=unit_tanh(kick_strength),
            # kick_strength=min(max(0., kick_strength), 1.),
        )


class SoccerAgentModel(nn.Module):
    obs_dim: int

    def __init__(self, obs_dim: int):
        super().__init__()

        self.obs_dim = obs_dim

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 3),
            # nn.Linear(11, 7),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Sigmoid(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Sigmoid(),
            # nn.Linear(11, 3),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
