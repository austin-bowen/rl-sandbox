import random
from math import atan2, sqrt
from typing import NamedTuple, Tuple

import numpy as np

from rlsandbox.agents.agent import Agent
from rlsandbox.envs.soccer import SoccerState, SoccerAction, Agent as EnvAgent
from rlsandbox.math import trunc_angle
from rlsandbox.types import Location2D


class SoccerAgent(Agent):
    def get_action(self, state: SoccerState) -> SoccerAction:
        obs = self._build_observation(state)

        return SoccerAction(
            move_dist=random.random(),
            # turn_angle=0.,
            turn_angle=0.2 * obs.ball_angle,
            # kick_strength=1.,
            kick_strength=random.random(),
        )

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
            ball_angle=ball_angle,
            ball_dist=ball_dist,
            ball_speed=ball_speed,
            ball_speed_angle=ball_speed_angle,
            left_goal_post_angle=left_goal_post_angle,
            left_goal_post_dist=left_goal_post_dist,
            right_goal_post_angle=right_goal_post_angle,
            right_goal_post_dist=right_goal_post_dist,
        )

    def _get_angle_and_dist(self, agent: EnvAgent, location: Location2D) -> Tuple[float, float]:
        dx = location.x - agent.location.x
        dy = location.y - agent.location.y

        angle = atan2(dy, dx) - agent.heading
        angle = trunc_angle(angle)
        dist = sqrt(dx ** 2 + dy ** 2)

        return angle, dist


class Observation(NamedTuple):
    ball_angle: float
    ball_dist: float
    ball_speed: float
    ball_speed_angle: float

    left_goal_post_angle: float
    left_goal_post_dist: float
    right_goal_post_angle: float
    right_goal_post_dist: float

    def to_numpy(self) -> np.ndarray:
        return np.array(self)
