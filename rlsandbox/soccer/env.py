from copy import deepcopy
from dataclasses import dataclass
from math import cos, sin, atan2, pi
from random import Random

import numpy as np

from rlsandbox.base import StateChange
from rlsandbox.base.env import Env
from rlsandbox.types import Size2D, Location2D, Velocity2D


@dataclass
class SoccerState:
    field_size: Size2D
    agent: 'SoccerAgent'
    ball: 'Ball'
    goal: 'Goal'
    steps: int = 0
    total_reward: float = np.zeros(2)


@dataclass
class SoccerAgent:
    location: Location2D
    heading: float


@dataclass
class Ball:
    location: Location2D
    velocity: Velocity2D
    speed_decay: float = 0.9


@dataclass
class Goal:
    left_post_location: Location2D
    right_post_location: Location2D


@dataclass
class SoccerAction:
    move_dist: float
    """Range: [-1, 1]"""

    turn_angle: float
    """Range: [-pi, pi]"""

    kick_strength: float
    """Range: [0, 1]"""

    def __str__(self) -> str:
        return f'(move_dist={self.move_dist:.3f}, ' \
               f'turn_angle={self.turn_angle:.3f}, ' \
               f'kick_strength={self.kick_strength:.3f})'


class SoccerEnv(Env):
    field_size: Size2D
    max_steps: int
    goal_reward: float
    step_reward: float
    kick_reward: float
    max_dist_to_ball: float
    max_ball_speed: float
    rng: Random

    _state: SoccerState

    def __init__(
            self,
            field_size: Size2D,
            max_steps: int,
            goal_reward: float = 10.,
            step_reward: float = -0.01,
            kick_reward: float = 0.1,
            max_dist_to_ball: float = 1.,
            max_ball_speed: float = 2.,
            rng: Random = None
    ):
        self.field_size = field_size
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.kick_reward = kick_reward
        self.max_dist_to_ball = max_dist_to_ball
        self.max_ball_speed = max_ball_speed
        self.rng = rng or Random()

        self.reset()

    def get_state(self) -> SoccerState:
        return self._state

    def reset(self) -> SoccerState:
        field_center = self.field_size.center

        goal_width = self.field_size.height / 3

        self._state = SoccerState(
            field_size=self.field_size,
            agent=SoccerAgent(
                location=Location2D(
                    # x=field_center.x,
                    x=self.rng.uniform(field_center.x / 2, field_center.x),
                    # y=field_center.y,
                    # y=self.rng.uniform(0, self.field_size.height),
                    y=self.rng.uniform(field_center.y - 3, field_center.y + 3),
                ),
                # heading=0.,
                heading=self.rng.uniform(-pi, pi),
            ),
            ball=Ball(
                location=Location2D(
                    # x=field_center.x * 1.5,
                    # y=field_center.y,
                    x=self.rng.uniform(0, self.field_size.width - 0.5),
                    y=self.rng.uniform(0, self.field_size.height),
                ),
                velocity=Velocity2D.zero(),
            ),
            goal=Goal(
                left_post_location=Location2D(
                    x=self.field_size.width,
                    y=field_center.y + goal_width / 2,
                ),
                right_post_location=Location2D(
                    x=self.field_size.width,
                    y=field_center.y - goal_width / 2,
                ),
            ),
        )

        return self._state

    def step(self, action: SoccerAction) -> StateChange:
        prev_state = deepcopy(self._state)

        self._simulate(action)

        reward = self._get_reward(prev_state, action)

        self._state.steps += 1
        self._state.total_reward += reward
        done = self._is_done()

        result = StateChange(
            state=prev_state,
            action=action,
            reward=reward,
            next_state=self._state,
            done=done,
        )

        return result

    def _simulate(self, action: SoccerAction) -> None:
        self._move_agent(action)
        self._move_ball(action)

    def _move_agent(self, action: SoccerAction) -> None:
        agent = self._state.agent

        agent.heading += action.turn_angle
        agent.location.x += action.move_dist * cos(agent.heading)
        agent.location.y += action.move_dist * sin(agent.heading)

        agent.location.x = min(max(0., agent.location.x), self.field_size.width)
        agent.location.y = min(max(0., agent.location.y), self.field_size.height)

    def _move_ball(self, action: SoccerAction) -> None:
        ball = self._state.ball

        if self._agent_is_near_ball():
            ball_speed = self.max_ball_speed * action.kick_strength

            agent = self._state.agent
            angle_from_agent_to_ball = atan2(
                ball.location.y - agent.location.y,
                ball.location.x - agent.location.x,
            )

            ball.velocity = Velocity2D(
                dx=ball_speed * cos(angle_from_agent_to_ball),
                dy=ball_speed * sin(angle_from_agent_to_ball),
            )

        ball.location.x += ball.velocity.dx
        ball.location.y += ball.velocity.dy

        if ball.location.x < 0:
            ball.location.x *= -1
            ball.velocity.dx *= -1
        elif ball.location.x > self.field_size.width:
            ball.location.x = 2 * self.field_size.width - ball.location.x
            ball.velocity.dx *= -1
        if ball.location.y < 0:
            ball.location.y *= -1
            ball.velocity.dy *= -1
        elif ball.location.y > self.field_size.height:
            ball.location.y = 2 * self.field_size.height - ball.location.y
            ball.velocity.dy *= -1

        ball.velocity.dx *= ball.speed_decay
        ball.velocity.dy *= ball.speed_decay

    def _agent_is_near_ball(self) -> bool:
        agent = self._state.agent
        ball = self._state.ball
        dist = (agent.location.x - ball.location.x) ** 2 + (agent.location.y - ball.location.y) ** 2

        return dist <= self.max_dist_to_ball ** 2

    def _get_reward(self, prev_state: SoccerState, action: SoccerAction) -> float:
        reward = np.zeros(2)

        reward[0] += self.step_reward

        if self._ball_is_in_goal():
            reward[0] += self.goal_reward
        # elif self._ball_is_out_of_bounds():
        #     reward -= self.goal_reward

        if self._state.ball.velocity.magnitude < 0.001:
            prev_dist_to_ball = prev_state.agent.location.euclidean_dist(prev_state.ball.location)
            curr_dist_to_ball = self._state.agent.location.euclidean_dist(self._state.ball.location)
            diff_dist_to_ball = prev_dist_to_ball - curr_dist_to_ball

            # if diff_dist_to_ball < 0:
            #     diff_dist_to_ball *= 2

            reward[1] += diff_dist_to_ball

        if action.kick_strength > 0 and self._agent_is_near_ball():
            reward[1] += self.kick_reward * action.kick_strength
            # reward[1] += self.kick_reward * (1 + action.kick_strength) ** 2
        # # TODO REMOVE THIS?
        # elif action.kick_strength > 0:
        #     reward -= action.kick_strength * 0.1

        goal_center = Location2D(x=self.field_size.width, y=self.field_size.height / 2)
        prev_dist_ball_to_goal = prev_state.ball.location.euclidean_dist(goal_center)
        curr_dist_ball_to_goal = self._state.ball.location.euclidean_dist(goal_center)
        diff_dist_ball_to_goal = prev_dist_ball_to_goal - curr_dist_ball_to_goal
        reward[1] += diff_dist_ball_to_goal * (1 if diff_dist_ball_to_goal > 0 else .5)

        reward[1] -= abs(action.turn_angle) * 0.05

        return reward

    def _is_done(self) -> bool:
        return (
                self._state.steps >= self.max_steps
                or self._ball_is_out_of_bounds()
                or self._ball_is_in_goal()
        )

    def _ball_is_out_of_bounds(self) -> bool:
        ball = self._state.ball

        return (
                ball.location.x < 0
                or ball.location.x > self.field_size.width
                or ball.location.y < 0
                or ball.location.y > self.field_size.height
        )

    def _ball_is_in_goal(self) -> bool:
        ball = self._state.ball
        goal = self._state.goal

        # TODO Make this better
        return (
                ball.location.x >= self.field_size.width - 0.5
                and goal.left_post_location.y >= ball.location.y >= goal.right_post_location.y
        )
