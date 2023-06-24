from math import cos, sin

from rlsandbox.envs.renderers.pygame_2d_renderer import Pygame2DEnvRenderer, BLACK, WHITE, BLUE
from rlsandbox.envs.soccer import SoccerEnv, SoccerState, Ball, SoccerAgent, SoccerAction
from rlsandbox.types import Location2D, Size2D, StateChange

SOCCER_FIELD_GREEN = (100, 200, 50)


class SoccerEnvRenderer(Pygame2DEnvRenderer):
    env: SoccerEnv

    def __init__(self, env: SoccerEnv, scale: float = 20., **kwargs):
        canvas_size = Size2D(
            width=env.field_size.width * scale,
            height=env.field_size.height * scale,
        )

        super().__init__(canvas_size=canvas_size, scale=scale, **kwargs)
        self.env = env

    def draw_state(self, state: SoccerState) -> None:
        self._draw_field(state)
        self._draw_agent(state.agent)
        self._draw_ball(state.ball)

    def draw_state_change(self, state_change: StateChange[SoccerState, SoccerAction]) -> None:
        self.draw_state(state_change.next_state)

    def _draw_field(self, state: SoccerState) -> None:
        self.canvas.fill(SOCCER_FIELD_GREEN)

        self._draw_goal_post(state.goal.left_post_location)
        self._draw_goal_post(state.goal.right_post_location)

    def _draw_goal_post(self, location: Location2D) -> None:
        self._draw_circle(BLACK, location, 0.25)

    def _draw_ball(self, ball: Ball) -> None:
        self._draw_circle(WHITE, ball.location, 0.25)

    def _draw_agent(self, agent: SoccerAgent) -> None:
        agent_radius = self.env.max_dist_to_ball
        self._draw_circle(BLUE, agent.location, agent_radius)

        point0 = agent.location
        point1 = Location2D(
            agent.location.x + agent_radius * cos(agent.heading),
            agent.location.y + agent_radius * sin(agent.heading),
        )
        self._draw_line(WHITE, point0, point1, width=0.1)
