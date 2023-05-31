from copy import deepcopy
from dataclasses import dataclass
from math import cos, sin, atan2
from random import Random

from rlsandbox.envs.env import Env
from rlsandbox.types import StateChange, Size2D, Location2D, Velocity2D


@dataclass
class SoccerState:
    field_size: Size2D
    agent: 'Agent'
    ball: 'Ball'
    goal: 'Goal'
    steps: int


@dataclass
class Agent:
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


class SoccerEnv(Env):
    field_size: Size2D
    max_steps: int
    max_dist_to_ball: float
    rng: Random

    _state: SoccerState

    def __init__(self, field_size: Size2D, max_steps: int, max_dist_to_ball: float = 1., rng: Random = None):
        self.field_size = field_size
        self.max_steps = max_steps
        self.max_dist_to_ball = max_dist_to_ball
        self.rng = rng or Random()

        self.reset()

    def reset(self) -> SoccerState:
        field_center = self.field_size.center

        goal_width = self.field_size.height / 3

        self._state = SoccerState(
            field_size=self.field_size,
            agent=Agent(
                location=Location2D(
                    x=self.rng.uniform(0, self.field_size.width),
                    y=self.rng.uniform(0, self.field_size.height),
                ),
                heading=0.,
            ),
            ball=Ball(
                location=Location2D(
                    x=field_center.x * 1.5,
                    y=field_center.y,
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
            steps=0,
        )

        return self._state

    def step(self, action: SoccerAction) -> StateChange:
        prev_state = deepcopy(self._state)

        self._simulate(action)

        reward = self._get_reward()

        self._state.steps += 1

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
            agent = self._state.agent

            angle_from_agent_to_ball = atan2(
                ball.location.y - agent.location.y,
                ball.location.x - agent.location.x,
            )

            ball.velocity = Velocity2D(
                dx=action.kick_strength * cos(angle_from_agent_to_ball),
                dy=action.kick_strength * sin(angle_from_agent_to_ball),
            )

        ball.location.x += ball.velocity.dx
        ball.location.y += ball.velocity.dy

        ball.velocity.dx *= ball.speed_decay
        ball.velocity.dy *= ball.speed_decay

    def _agent_is_near_ball(self) -> bool:
        agent = self._state.agent
        ball = self._state.ball
        dist = (agent.location.x - ball.location.x) ** 2 + (agent.location.y - ball.location.y) ** 2

        return dist <= self.max_dist_to_ball ** 2

    def _get_reward(self) -> float:
        # TODO Make this better
        return 1. if self._ball_is_in_goal() else 0.

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
                ball.location.x >= self.field_size.width
                and goal.left_post_location.y >= ball.location.y >= goal.right_post_location.y
        )
