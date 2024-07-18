from typing import Optional, Iterable


class CircularList(list):
    def __init__(self, items: Iterable = None, max_size: int = None):
        super().__init__()

        self._max_size = max_size
        self._next_index = 0

        self._is_max_size = (
            (lambda: False) if max_size is None else
            (lambda: len(self) >= max_size)
        )

        if items is not None:
            self.extend(items)

    @property
    def is_max_size(self) -> bool:
        return self._is_max_size()

    @property
    def max_size(self) -> Optional[int]:
        return self._max_size

    def append(self, item) -> None:
        if self.is_max_size:
            self[self._next_index] = item
            self._next_index = (self._next_index + 1) % self.max_size
        else:
            super().append(item)

    def extend(self, items) -> None:
        for item in items:
            self.append(item)

    def insert(self, index, item) -> None:
        raise NotImplementedError()

    def remove(self, value) -> None:
        raise NotImplementedError()

    def pop(self, index: int = -1) -> None:
        raise NotImplementedError()

    def clear(self) -> None:
        super().clear()
        self._next_index = 0

    def copy(self) -> 'CircularList':
        return CircularList(self, max_size=self.max_size)
