import numpy as np
from typing import List, Dict


class WiSARD:

    def __init__(self, input_size: int, tuple_size: int, n_classes: int):
        if input_size % tuple_size != 0:
            input_size = (input_size // tuple_size) * tuple_size
            self.truncated = True
        else:
            self.truncated = False

        self.input_size = input_size
        self.tuple_size = tuple_size
        self.n_rams = input_size // tuple_size
        self.n_classes = n_classes

        self.rams: List[List[Dict[int, int]]] = [
            [dict() for _ in range(self.n_rams)] for _ in range(self.n_classes)
        ]

    def _sample_to_chunks(self, x: np.ndarray):

        if self.truncated:
            x = x[: self.input_size]

        chunks = x.reshape(self.n_rams, self.tuple_size)
        powers = 1 << np.arange(self.tuple_size, dtype=np.uint64)
        addresses = chunks.dot(powers)
        return addresses  

    def fit(self, X: np.ndarray, y: np.ndarray):

        X = X.astype(np.uint8)
        for xi, label in zip(X, y):
            addresses = self._sample_to_chunks(xi)
            for ram_idx, addr in enumerate(addresses):
                ram = self.rams[label][ram_idx]
                ram[addr] = ram.get(addr, 0) + 1

    def predict(self, X: np.ndarray) -> np.ndarray:

        X = X.astype(np.uint8)
        preds = []
        for xi in X:
            addresses = self._sample_to_chunks(xi)
            scores = np.zeros(self.n_classes, dtype=np.int32)

            for c in range(self.n_classes):
                for ram_idx, addr in enumerate(addresses):
                    if addr in self.rams[c][ram_idx]:
                        scores[c] += self.rams[c][ram_idx][addr]

            preds.append(int(np.argmax(scores)))

        return np.array(preds, dtype=int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:

        y_pred = self.predict(X)
        return (y_pred == y).mean()
