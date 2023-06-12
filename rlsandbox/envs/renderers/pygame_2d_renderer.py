from abc import abstractmethod
from typing import Tuple

import pygame

from rlsandbox.envs.env import Env
from rlsandbox.envs.renderers.renderer import EnvRenderer
from rlsandbox.types import Size2D, Location2D

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Pygame2DEnvRenderer(EnvRenderer):
    canvas_size: Size2D
    fps: int
    scale: float
    clear_color: Tuple

    _canvas: pygame.Surface = None
    _clock: pygame.time.Clock = None

    def __init__(
            self,
            canvas_size: Size2D,
            fps: int = 30,
            scale: float = 1.0,
            clear_color: Tuple = WHITE,
    ):
        self.canvas_size = canvas_size
        self.fps = fps
        self.scale = scale
        self.clear_color = clear_color

        self._clock = pygame.time.Clock()

    @property
    def canvas(self) -> pygame.Surface:
        if self._canvas is None:
            pygame.init()
            self._canvas = pygame.display.set_mode((self.canvas_size.width, self.canvas_size.height))

        return self._canvas

    def render(self, env: Env) -> None:
        self._clear()
        self.draw_env(env)
        pygame.display.flip()

        for _ in pygame.event.get():
            pass

        self._clock.tick(self.fps)

    @abstractmethod
    def draw_env(self, env: Env) -> None:
        ...

    def _clear(self) -> None:
        self.canvas.fill(self.clear_color)

    def _draw_circle(self, color: Tuple, location: Location2D, radius: float) -> None:
        pygame.draw.circle(self.canvas, color, self._location_to_pygame(location), round(radius * self.scale))

    def _draw_line(self, color: Tuple, point0: Location2D, point1: Location2D, width: float = 1.) -> None:
        pygame.draw.line(
            self.canvas,
            color,
            self._location_to_pygame(point0),
            self._location_to_pygame(point1),
            round(width * self.scale),
        )

    def _location_to_pygame(self, location: Location2D) -> Tuple:
        return (
            round(location.x * self.scale),
            round(self.canvas_size.height - (location.y * self.scale)),
        )
