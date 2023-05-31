from math import pi


def trunc_angle(angle: float) -> float:
    """
    Truncates the angle such that it is kept in range ``(-pi, pi]``.

    Examples::

        0      --> 0
        pi     --> pi
        -pi    --> pi
        pi/2   --> pi/2
        3 pi/2 --> -pi/2
        5 pi   --> pi
        1.1 pi --> -0.9 pi
    """

    # Shift the angle up
    angle += pi

    # Truncate the angle to range (0, 2 * pi]
    #     mod( x, y) --> [ 0, y)
    #     mod(-x, y) --> ( y, 0]
    #   - mod(-x, y) --> (-y, 0]
    # y - mod(-x, y) --> (0, y]
    two_pi = 2 * pi
    angle = two_pi - (-angle % two_pi)

    # Shift the angle back down
    angle -= pi

    return angle
