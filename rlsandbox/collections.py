from random import Random


class ReservoirSampledList:
    """
    A list with a maximum length that samples items uniformly at random
    from all items appended.
    """

    def __init__(self, max_len: int, rng: Random = None):
        self.max_len = max_len
        self.data = []
        self.rng = rng or Random()
        self._append_count = 0

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def append(self, item) -> None:
        if len(self) < self.max_len:
            self.data.append(item)
            return

        idx = self.rng.randint(0, self._append_count)
        if idx < self.max_len:
            self.data[idx] = item

        self._append_count += 1

    def extend(self, iterable) -> None:
        for item in iterable:
            self.append(item)
