from dataclasses import dataclass
import operator
from typing import Any

Action = Any
Reward = float
State = Any


@dataclass
class StateChange:
    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool


@dataclass
class Vector2D:
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other) -> 'Vector2D':
        return self._apply_operator(operator.add, other)

    def __radd__(self, other) -> 'Vector2D':
        return self + other

    def __sub__(self, other) -> 'Vector2D':
        return self._apply_operator(operator.sub, other)

    def __rsub__(self, other) -> 'Vector2D':
        return Vector2D(other - self.x, other - self.y)

    def __mul__(self, other) -> 'Vector2D':
        return self._apply_operator(operator.mul, other)

    def __rmul__(self, other) -> 'Vector2D':
        return self * other

    def __truediv__(self, other) -> 'Vector2D':
        return self._apply_operator(operator.truediv, other)

    def __rtruediv__(self, other) -> 'Vector2D':
        return Vector2D(other / self.x, other / self.y)

    def _apply_operator(self, op, other) -> 'Vector2D':
        if isinstance(other, Vector2D):
            return Vector2D(op(self.x, other.x), op(self.y, other.y))
        else:
            return Vector2D(op(self.x, other), op(self.y, other))


Location2D = Vector2D
Point2D = Vector2D


@dataclass
class Velocity2D:
    dx: float
    dy: float

    @staticmethod
    def zero() -> 'Velocity2D':
        return Velocity2D(0., 0.)


@dataclass
class Size2D:
    width: float
    height: float

    @property
    def center(self) -> 'Location2D':
        return Location2D(self.width / 2, self.height / 2)


@dataclass
class Orientation2D:
    x: float
    y: float
    theta: float
