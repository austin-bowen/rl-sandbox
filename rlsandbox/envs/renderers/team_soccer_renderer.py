from math import cos, sin

from rlsandbox.envs.renderers.pygame_2d_renderer import Pygame2DEnvRenderer, WHITE, BLUE, RED
from rlsandbox.envs.soccer import Ball
from rlsandbox.envs.team_soccer import TeamSoccerAgent, TeamSoccerEnv, TeamSoccerState, TeamId
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

    def draw_env(self, env: TeamSoccerEnv) -> None:
        state = env.get_state()
        self._draw_field(state)
        self._draw_agents(state.agents)
        self._draw_ball(state.ball)

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


def main():
    env = TeamSoccerEnv(
        field_size=Size2D(width=40, height=20),
        left_team_size=3,
        right_team_size=3,
        max_steps=100,
    )

    renderer = TeamSoccerEnvRenderer(env=env, scale=20)
    renderer.render(env)

    input('Press enter to exit')


if __name__ == '__main__':
    main()
