def assert_shape(array, shape) -> None:
    assert array.shape == shape, (array.shape, shape)


def every(n: int, i: int) -> bool:
    return (i + 1) % n == 0


def printne(*args, **kwargs) -> None:
    """print with no ending newline character."""
    print(*args, end='', flush=True, **kwargs)


class Wrapper:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        return getattr(self.obj, name)
