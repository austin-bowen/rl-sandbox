from abc import abstractmethod

import numpy as np
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor


class KnnRegressor:
    @abstractmethod
    def get_items(self) -> tuple[ndarray, ndarray]:
        ...

    def has_items(self) -> bool:
        return self.get_items()[0] is not None

    @abstractmethod
    def fit(self, input: ndarray, output: ndarray) -> None:
        ...

    @abstractmethod
    def add(self, input: ndarray, output: ndarray) -> None:
        ...

    @abstractmethod
    def remove(self, index: int) -> None:
        ...

    @abstractmethod
    def predict(self, input: ndarray) -> ndarray:
        ...


class SklearnKnnRegressor(KnnRegressor):
    def __init__(self, max_items: int = None, **kwargs):
        super().__init__()

        self.max_items = max_items
        self._kwargs = kwargs

        self._knn = None
        self._inputs = None
        self._outputs = None

    def get_items(self) -> tuple[ndarray, ndarray]:
        return self._inputs, self._outputs

    def fit(self, input: ndarray, output: ndarray) -> None:
        if self.max_items is not None and input.shape[0] > self.max_items:
            input = input[-self.max_items:]
            output = output[-self.max_items:]

        self._inputs = input
        self._outputs = output

        self._knn = KNeighborsRegressor(**self._kwargs)
        self._knn.fit(self._inputs, self._outputs)

    def add(self, input: ndarray, output: ndarray) -> None:
        if self._inputs is not None:
            input = np.vstack([self._inputs, input])
            output = np.concatenate([self._outputs, output])

        self.fit(input, output)

    def remove(self, index: int | list[int]) -> None:
        np.delete(self._inputs, index, axis=0)
        np.delete(self._outputs, index, axis=0)

        self.fit(self._inputs, self._outputs)

    def predict(self, input: ndarray) -> ndarray:
        return self._knn.predict(input)
