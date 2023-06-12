from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import cos, sin, atan2, pi
from random import Random
from typing import List, Optional, Tuple, Dict

import numpy as np

from rlsandbox.envs.env import Env
from rlsandbox.envs.soccer import Ball, SoccerAction
from rlsandbox.types import StateChange, Size2D, Location2D, Velocity2D


class TeamId(Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass
class TeamSoccerState:
    field_size: Size2D
    left_team_agents: List['TeamSoccerAgent']
    right_team_agents: List['TeamSoccerAgent']
    ball: 'Ball'
    left_goal: 'Goal'
    right_goal: 'Goal'
    steps: int

    @property
    def agents(self) -> List['TeamSoccerAgent']:
        return self.left_team_agents + self.right_team_agents

    @property
    def agent_ids(self) -> List['AgentId']:
        return [agent.id for agent in self.agents]

    def agent_with_id(self, agent_id: 'AgentId') -> 'TeamSoccerAgent':
        for agent in self.agents:
            if agent.id == agent_id:
                return agent

        raise ValueError(f'Agent with id {agent_id!r} not found')


@dataclass
class TeamSoccerAgent:
    id: 'AgentId'
    location: Location2D
    heading: float


@dataclass(frozen=True)
class AgentId:
    team: TeamId
    number: int


@dataclass
class Goal:
    left_post_location: Location2D
    right_post_location: Location2D


SoccerActions = Dict[AgentId, SoccerAction]
Rewards = Dict[AgentId, float]


class TeamSoccerEnv(Env):
    field_size: Size2D
    left_team_size: int
    right_team_size: int
    max_steps: int
    goal_reward: float
    step_reward: float
    kick_reward: float
    max_dist_to_ball: float
    rng: Random

    _state: TeamSoccerState

    def __init__(
            self,
            field_size: Size2D,
            left_team_size: int,
            right_team_size: int,
            max_steps: int,
            goal_reward: float = 10.,
            step_reward: float = -0.01,
            kick_reward: float = 0.1,
            max_dist_to_ball: float = 1.,
            rng: Random = None
    ):
        assert left_team_size > 0, left_team_size
        assert right_team_size > 0, right_team_size

        self.left_team_size = left_team_size
        self.right_team_size = right_team_size
        self.field_size = field_size
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.kick_reward = kick_reward
        self.max_dist_to_ball = max_dist_to_ball
        self.rng = rng or Random()

        self.reset()

    def get_state(self) -> TeamSoccerState:
        return self._state

    def reset(self) -> TeamSoccerState:
        field_center = self.field_size.center

        goal_width = self.field_size.height / 3

        self._state = TeamSoccerState(
            field_size=self.field_size,
            left_team_agents=self._team_generator(TeamId.LEFT, self.left_team_size),
            right_team_agents=self._team_generator(TeamId.RIGHT, self.right_team_size),
            ball=Ball(
                location=Location2D(
                    # x=field_center.x * 1.5,
                    # y=field_center.y,
                    x=self.rng.uniform(0, self.field_size.width - 0.5),
                    y=self.rng.uniform(0, self.field_size.height),
                ),
                velocity=Velocity2D.zero(),
            ),
            left_goal=Goal(
                left_post_location=Location2D(
                    x=0,
                    y=field_center.y - goal_width / 2,
                ),
                right_post_location=Location2D(
                    x=0,
                    y=field_center.y + goal_width / 2,
                ),
            ),
            right_goal=Goal(
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

    def _team_generator(self, team: TeamId, size: int) -> List[TeamSoccerAgent]:
        return [
            TeamSoccerAgent(
                id=AgentId(team, number),
                location=Location2D(
                    x=self.rng.uniform(0, self.field_size.width),
                    y=self.rng.uniform(0, self.field_size.height),
                ),
                heading=self.rng.uniform(-pi, pi),
            )
            for number in range(size)
        ]

    def step(self, actions: SoccerActions) -> StateChange:
        self._check_actions(actions)

        prev_state = deepcopy(self._state)

        self._simulate(actions)

        rewards = self._get_rewards(prev_state, actions)

        self._state.steps += 1

        done = self._is_done()

        result = StateChange(
            state=prev_state,
            action=actions,
            reward=rewards,
            next_state=self._state,
            done=done,
        )

        return result

    def _check_actions(self, actions: SoccerActions) -> None:
        actions_agent_ids = set(actions.keys())
        expected_agent_ids = set(self._state.agent_ids)
        if actions_agent_ids != expected_agent_ids:
            raise ValueError(
                f'Invalid agent IDs; '
                f'Action IDs: {actions_agent_ids}; '
                f'Expected IDs: {expected_agent_ids}'
            )

    def _simulate(self, actions: SoccerActions) -> None:
        self._move_agents(actions)
        self._move_ball(actions)

    def _move_agents(self, actions: SoccerActions) -> None:
        for agent_id, action in actions.items():
            self._move_agent(agent_id, action)

    def _move_agent(self, agent_id: AgentId, action: SoccerAction) -> None:
        agent = self._state.agent_with_id(agent_id)

        agent.heading += action.turn_angle
        agent.location.x += action.move_dist * cos(agent.heading)
        agent.location.y += action.move_dist * sin(agent.heading)

        agent.location.x = min(max(0., agent.location.x), self.field_size.width)
        agent.location.y = min(max(0., agent.location.y), self.field_size.height)

    def _move_ball(self, actions: SoccerActions) -> None:
        ball = self._state.ball

        kick_agent, kick_strength = self._get_kick(actions)
        if kick_agent is not None:
            angle_from_agent_to_ball = atan2(
                ball.location.y - kick_agent.location.y,
                ball.location.x - kick_agent.location.x,
            )

            ball.velocity = Velocity2D(
                dx=kick_strength * cos(angle_from_agent_to_ball),
                dy=kick_strength * sin(angle_from_agent_to_ball),
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

    def _get_kick(self, actions: SoccerActions) -> Tuple[Optional[TeamSoccerAgent], float]:
        kicking_agents: List[Tuple[TeamSoccerAgent, float]] = []

        for agent_id, action in actions.items():
            agent = self._state.agent_with_id(agent_id)
            kick_strength = action.kick_strength

            if kick_strength > 0 and self._agent_is_near_ball(agent):
                kicking_agents.append((agent, kick_strength))

        return self.rng.choice(kicking_agents) if kicking_agents else (None, 0)

    def _agent_is_near_ball(self, agent: TeamSoccerAgent) -> bool:
        ball = self._state.ball
        dist = (agent.location.x - ball.location.x) ** 2 + (agent.location.y - ball.location.y) ** 2

        return dist <= self.max_dist_to_ball ** 2

    def _get_rewards(self, prev_state: TeamSoccerState, actions: SoccerActions) -> Rewards:
        return {
            agent_id: self._get_reward(prev_state, agent_id, action)
            for agent_id, action in actions.items()
        }

    def _get_reward(self, prev_state: TeamSoccerState, agent_id: AgentId, action: SoccerAction) -> float:
        agent = self._state.agent_with_id(agent_id)
        reward = np.zeros(2)

        reward[0] += self.step_reward

        if self._ball_is_in_opponent_goal(agent):
            reward[0] += self.goal_reward
        elif self._ball_is_in_team_goal(agent):
            reward[0] -= self.goal_reward

        if self._state.ball.velocity.magnitude < 0.001:
            prev_agent = prev_state.agent_with_id(agent.id)
            curr_agent = self._state.agent_with_id(agent.id)

            prev_dist_to_ball = prev_agent.location.euclidean_dist(prev_state.ball.location)
            curr_dist_to_ball = curr_agent.location.euclidean_dist(self._state.ball.location)
            diff_dist_to_ball = prev_dist_to_ball - curr_dist_to_ball

            # if diff_dist_to_ball < 0:
            #     diff_dist_to_ball *= 2

            reward[1] += diff_dist_to_ball

        if action.kick_strength > 0 and self._agent_is_near_ball(agent):
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
                or self._ball_is_in_goal(TeamId.LEFT)
                or self._ball_is_in_goal(TeamId.RIGHT)
        )

    def _ball_is_out_of_bounds(self) -> bool:
        ball = self._state.ball

        return (
                ball.location.x < 0
                or ball.location.x > self.field_size.width
                or ball.location.y < 0
                or ball.location.y > self.field_size.height
        )

    def _ball_is_in_opponent_goal(self, agent: TeamSoccerAgent) -> bool:
        opponent_team = TeamId.LEFT if agent.id.team == TeamId.RIGHT else TeamId.RIGHT
        return self._ball_is_in_goal(opponent_team)

    def _ball_is_in_team_goal(self, agent: TeamSoccerAgent) -> bool:
        return self._ball_is_in_goal(agent.id.team)

    def _ball_is_in_goal(self, team: TeamId) -> bool:
        ball = self._state.ball

        # TODO Make this better
        goal_depth = 0.5
        if team == TeamId.LEFT:
            goal = self._state.left_goal
            return (
                    ball.location.x <= goal_depth
                    and goal.left_post_location.y <= ball.location.y <= goal.right_post_location.y
            )
        else:
            goal = self._state.right_goal
            return (
                    ball.location.x >= self.field_size.width - goal_depth
                    and goal.left_post_location.y >= ball.location.y >= goal.right_post_location.y
            )
