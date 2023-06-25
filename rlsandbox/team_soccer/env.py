from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import cos, sin, atan2, pi
from random import Random
from typing import List, Optional, Tuple, Dict

import numpy as np
from shapely import LineString

from rlsandbox.base import Env, StateChange
from rlsandbox.soccer.env import Ball, SoccerAction
from rlsandbox.types import Size2D, Location2D, Velocity2D


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
    last_kicker: Optional['AgentId'] = None
    steps: int = 0
    steps_no_ball_movement: int = 0
    done: bool = False

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
    max_steps: Optional[int]
    max_steps_no_ball_movement: Optional[int]
    goal_reward: float
    step_reward: float
    kick_reward: float
    max_ball_speed: float
    max_dist_to_ball: float
    rng: Random

    _state: TeamSoccerState

    def __init__(
            self,
            field_size: Size2D,
            left_team_size: int,
            right_team_size: int,
            max_steps: int = None,
            max_steps_no_ball_movement: int = None,
            goal_reward: float = 10.,
            step_reward: float = -0.01,
            kick_reward: float = 0.,
            max_ball_speed: float = 3.,
            max_dist_to_ball: float = 1.,
            rng: Random = None
    ):
        assert left_team_size > 0, left_team_size
        assert right_team_size > 0, right_team_size

        self.left_team_size = left_team_size
        self.right_team_size = right_team_size
        self.field_size = field_size
        self.max_steps = max_steps
        self.max_steps_no_ball_movement = max_steps_no_ball_movement
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.kick_reward = kick_reward
        self.max_ball_speed = max_ball_speed
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
        self._state.done = self._is_done(prev_state)

        result = StateChange(
            state=prev_state,
            action=actions,
            reward=rewards,
            next_state=self._state,
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
            ball_speed = self.max_ball_speed * kick_strength

            angle_from_agent_to_ball = atan2(
                ball.location.y - kick_agent.location.y,
                ball.location.x - kick_agent.location.x,
            )

            ball.velocity = Velocity2D(
                dx=ball_speed * cos(angle_from_agent_to_ball),
                dy=ball_speed * sin(angle_from_agent_to_ball),
            )

            self._state.last_kicker = kick_agent.id

        prev_ball_location = deepcopy(ball.location)

        ball.location.x += ball.velocity.dx
        ball.location.y += ball.velocity.dy

        ball_is_in_goal = (
                self._ball_is_in_goal(ball.location, prev_ball_location, self._state.left_goal)
                or self._ball_is_in_goal(ball.location, prev_ball_location, self._state.right_goal)
        )

        if not ball_is_in_goal:
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

        if ball.velocity.magnitude < 0.001:
            self._state.steps_no_ball_movement += 1
        else:
            self._state.steps_no_ball_movement = 0

    def _get_kick(self, actions: SoccerActions) -> Tuple[Optional[TeamSoccerAgent], float]:
        kicking_agents: List[Tuple[TeamSoccerAgent, float]] = []

        for agent_id, action in actions.items():
            agent = self._state.agent_with_id(agent_id)
            kick_strength = action.kick_strength

            if kick_strength > 0 and self._agent_is_near_ball(agent):
                kick_strength = min(kick_strength, 1)
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

        ball_location = self._state.ball.location
        prev_ball_location = prev_state.ball.location
        if self._ball_is_in_opponent_goal(ball_location, prev_ball_location, agent) \
                and self._state.last_kicker == agent_id:
            reward[0] += self.goal_reward
        elif self._ball_is_in_team_goal(ball_location, prev_ball_location, agent):
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

        opponent_goal_center = Location2D(
            x=0 if agent.id.team == TeamId.RIGHT else self.field_size.width,
            y=self.field_size.height / 2,
        )
        prev_dist_ball_to_goal = prev_state.ball.location.euclidean_dist(opponent_goal_center)
        curr_dist_ball_to_goal = self._state.ball.location.euclidean_dist(opponent_goal_center)
        dist_ball_closer_to_goal = prev_dist_ball_to_goal - curr_dist_ball_to_goal
        reward[1] += (
            dist_ball_closer_to_goal
            # if dist_ball_closer_to_goal >= 0
            # else 2 * dist_ball_closer_to_goal
        )

        reward[1] -= abs(action.turn_angle) * 0.1

        took_ball_reward = 1
        if self._took_ball_from_opponent(agent, prev_state):
            reward[1] += took_ball_reward
        if self._lost_ball_to_opponent(agent, prev_state):
            reward[1] -= took_ball_reward
        # if self._state.last_kicker and self._state.last_kicker.team != agent.id.team:
        #     reward[1] -= 0.1

        # return reward
        return reward[0] + reward[1]

    def _took_ball_from_opponent(self, agent: TeamSoccerAgent, prev_state: TeamSoccerState) -> bool:
        return (
                prev_state.last_kicker is not None
                and self._state.last_kicker == agent.id
                and prev_state.last_kicker.team != agent.id.team
        )

    def _lost_ball_to_opponent(self, agent: TeamSoccerAgent, prev_state: TeamSoccerState) -> bool:
        return (
                prev_state.last_kicker is not None
                and prev_state.last_kicker == agent.id
                and self._state.last_kicker.team != agent.id.team
        )

    def _is_done(self, prev_state: TeamSoccerState) -> bool:
        ball_location = self._state.ball.location
        prev_ball_location = prev_state.ball.location

        return (
                (self.max_steps and self._state.steps >= self.max_steps)
                or (self.max_steps_no_ball_movement
                    and self._state.steps_no_ball_movement >= self.max_steps_no_ball_movement)
                or self._ball_is_out_of_bounds()
                or self._ball_is_in_goal(ball_location, prev_ball_location, self._state.left_goal)
                or self._ball_is_in_goal(ball_location, prev_ball_location, self._state.right_goal)
        )

    def _ball_is_out_of_bounds(self) -> bool:
        ball = self._state.ball

        return (
                ball.location.x < 0
                or ball.location.x > self.field_size.width
                or ball.location.y < 0
                or ball.location.y > self.field_size.height
        )

    def _ball_is_in_opponent_goal(
            self,
            ball_location: Location2D,
            prev_ball_location: Location2D,
            agent: TeamSoccerAgent,
    ) -> bool:
        goal = self._state.right_goal if agent.id.team == TeamId.LEFT else self._state.left_goal
        return self._ball_is_in_goal(ball_location, prev_ball_location, goal)

    def _ball_is_in_team_goal(
            self,
            ball_location: Location2D,
            prev_ball_location: Location2D,
            agent: TeamSoccerAgent,
    ) -> bool:
        goal = self._state.left_goal if agent.id.team == TeamId.LEFT else self._state.right_goal
        return self._ball_is_in_goal(ball_location, prev_ball_location, goal)

    def _ball_is_in_goal(
            self,
            ball_location: Location2D,
            prev_ball_location: Location2D,
            goal: Goal,
    ) -> bool:
        ball_travel_line = LineString([
            tuple(prev_ball_location),
            tuple(ball_location),
        ])

        goal_line = LineString([
            tuple(goal.left_post_location),
            tuple(goal.right_post_location),
        ])

        return ball_travel_line.intersects(goal_line)
