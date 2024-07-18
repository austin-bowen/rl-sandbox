from typing import Optional, Iterable


class CircularList(list):
    def __init__(self, items: Iterable = None, max_len: int = None):
        super().__init__()

        self._max_len = max_len
        self._next_index = 0

        self._is_max_len = (
            (lambda: False) if max_len is None else
            (lambda: len(self) >= max_len)
        )

        if items is not None:
            self.extend(items)

    @property
    def is_max_len(self) -> bool:
        return self._is_max_len()

    @property
    def max_len(self) -> Optional[int]:
        return self._max_len

    def append(self, item) -> None:
        if self.is_max_len:
            self[self._next_index] = item
            self._next_index = (self._next_index + 1) % self.max_len
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
        return CircularList(self, max_len=self.max_len)
