from torch import Tensor


def assert_is_binary(tensor: Tensor) -> None:
    assert ((tensor == 0) | (tensor == 1)).all(), tensor


def zip_require_same_len(*iterables):
    iterables = [iter(i) for i in iterables]

    yield from zip(*iterables)

    for i, iterable in enumerate(iterables):
        try:
            next(iterable)
        except StopIteration:
            pass
        else:
            raise ValueError(f'Iterable {i} is longer than the shortest iterable')
