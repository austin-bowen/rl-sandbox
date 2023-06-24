import time
from math import cos, sin
from typing import Iterable

import pygame

from rlsandbox.base import Reward, StateChange
from rlsandbox.base.renderer.pygame_2d_renderer import Pygame2DEnvRenderer, WHITE, BLUE, RED
from rlsandbox.soccer.env import Ball, SoccerAction
from rlsandbox.team_soccer.env import TeamSoccerAgent, TeamSoccerEnv, TeamSoccerState, TeamId, SoccerActions, AgentId
from rlsandbox.types import Location2D, Size2D

SOCCER_FIELD_GREEN = (100, 200, 50)
LEFT_TEAM_COLOR = RED
RIGHT_TEAM_COLOR = BLUE


class TeamSoccerEnvRenderer(Pygame2DEnvRenderer):
    env: TeamSoccerEnv
    left_team_color: tuple
    right_team_color: tuple

    def __init__(
            self,
            env: TeamSoccerEnv,
            left_team_color: tuple = LEFT_TEAM_COLOR,
            right_team_color: tuple = RIGHT_TEAM_COLOR,
            scale: float = 20.,
            **kwargs,
    ):
        canvas_size = Size2D(
            width=env.field_size.width * scale,
            height=env.field_size.height * scale,
        )

        super().__init__(canvas_size=canvas_size, scale=scale, **kwargs)

        self.env = env
        self.left_team_color = left_team_color
        self.right_team_color = right_team_color

    def draw_state(self, state: TeamSoccerState) -> None:
        self._draw_field(state)
        self._draw_agents(state.agents)
        self._draw_ball(state.ball)

    def draw_state_change(self, state_change: StateChange[TeamSoccerState, SoccerActions]) -> None:
        self.draw_state(state_change.next_state)
        y = self._draw_actions_and_rewards(state_change)

        if state_change.state.done:
            time.sleep(3)

    def _draw_field(self, state: TeamSoccerState) -> None:
        self.canvas.fill(SOCCER_FIELD_GREEN)

        for goal, color in [
            (state.left_goal, self.left_team_color),
            (state.right_goal, self.right_team_color),
        ]:
            self._draw_goal_post(goal.left_post_location, color)
            self._draw_goal_post(goal.right_post_location, color)

    def _draw_goal_post(self, location: Location2D, color: tuple) -> None:
        self._draw_circle(color, location, 0.25)

    def _draw_ball(self, ball: Ball) -> None:
        self._draw_circle(WHITE, ball.location, 0.25)

    def _draw_agents(self, agents: list[TeamSoccerAgent]) -> None:
        for agent in agents:
            self._draw_agent(agent)

    def _draw_agent(self, agent: TeamSoccerAgent) -> None:
        agent_radius = self.env.max_dist_to_ball

        color = self.left_team_color if agent.id.team == TeamId.LEFT else self.right_team_color
        self._draw_circle(color, agent.location, agent_radius)

        point0 = agent.location
        point1 = Location2D(
            agent.location.x + agent_radius * cos(agent.heading),
            agent.location.y + agent_radius * sin(agent.heading),
        )
        self._draw_line(WHITE, point0, point1, width=0.1)

        self._draw_surface(
            self._build_text_surface(
                f'{agent.id.number}',
            ),
            agent.location + Location2D(x=agent_radius, y=-agent_radius),
        )

    def _draw_actions_and_rewards(self, state_change: StateChange) -> None:
        actions = state_change.action
        rewards = state_change.reward

        left_actions = [it for it in actions.items() if it[0].team == TeamId.LEFT]
        left_rewards = [it for it in rewards.items() if it[0].team == TeamId.LEFT]
        right_actions = [it for it in actions.items() if it[0].team == TeamId.RIGHT]
        right_rewards = [it for it in rewards.items() if it[0].team == TeamId.RIGHT]

        agent_number = lambda it: it[0].number
        left_actions.sort(key=agent_number)
        left_rewards.sort(key=agent_number)
        right_actions.sort(key=agent_number)
        right_rewards.sort(key=agent_number)

        last_kicker = state_change.next_state.last_kicker
        self._draw_messages_from_top_left(
            self._get_team_messages(
                'Left Team',
                left_actions,
                left_rewards,
                has_ball=last_kicker and last_kicker.team == TeamId.LEFT,
            ),
        )

        self._draw_messages_from_top_right(
            self._get_team_messages(
                'Right Team',
                right_actions,
                right_rewards,
                has_ball=last_kicker and last_kicker.team == TeamId.RIGHT,
            ),
        )

    def _get_team_messages(
            self,
            team: str,
            actions: list[tuple[AgentId, SoccerAction]],
            rewards: list[tuple[AgentId, Reward]],
            has_ball: bool,
    ) -> list:
        messages = [
            (team, dict(bold=True)),
            ('Actions:', dict(italic=True)),
        ]

        for agent_id, action in actions:
            messages.append((
                f'  {agent_id.number}: {action}',
                dict(),
            ))

        messages.append(('Rewards:', dict(italic=True)))
        for agent_id, reward in rewards:
            messages.append((
                f'  {agent_id.number}: {reward:.3f}',
                dict(),
            ))

        if has_ball:
            messages.append((
                f'Has ball!',
                dict(bold=True),
            ))

        return messages

    def _draw_messages_from_top_left(self, messages: Iterable[tuple[str, dict[str, str]]]) -> float:
        text_surfaces = self._messages_to_text_surfaces(messages)

        location = Location2D(x=0, y=self.env.field_size.height)
        for surface in text_surfaces:
            location.y -= surface.get_height() / self.scale
            self._draw_surface(surface, location)

        return location.y

    def _draw_messages_from_top_right(self, messages: Iterable[tuple[str, dict[str, str]]]) -> None:
        text_surfaces = self._messages_to_text_surfaces(messages)

        max_width = max(surface.get_width() for surface in text_surfaces)
        x = self.env.field_size.width - max_width / self.scale

        location = Location2D(x, y=self.env.field_size.height)
        for surface in text_surfaces:
            location.y -= surface.get_height() / self.scale
            self._draw_surface(surface, location)

    def _messages_to_text_surfaces(self, messages: Iterable[tuple[str, dict[str, str]]]) -> list[pygame.Surface]:
        return [
            self._build_text_surface(message, **kwargs)
            for message, kwargs in messages
        ]


def main():
    env = TeamSoccerEnv(
        field_size=Size2D(width=40, height=20),
        left_team_size=3,
        right_team_size=3,
        max_steps=100,
    )

    renderer = TeamSoccerEnvRenderer(env=env, scale=20)
    renderer.render_state(env.get_state())

    input('Press enter to exit')


if __name__ == '__main__':
    main()
