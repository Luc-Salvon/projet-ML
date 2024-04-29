import math
import numpy as np
from classes_abstraites import Module


class Linear(Module):
    def __init__(self, input: int, output: int):
        super().__init__()

        self.input_size = input
        self.output_size = output

        self._parameters = np.random.uniform(-math.sqrt(self.input_size), math.sqrt(self.input_size), (self.input_size + 1, self.output_size))
        self._gradient = np.zeros((self.input_size + 1, self.output_size))

    def zero_grad(self):
        self._gradient.fill(0)

    def forward(self, X: np.ndarray):
        assert X.shape[1] == self.input_size
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # Ajout du biais
        return X @ self._parameters

    def backward_update_gradient(self, input: np.ndarray, delta: np.ndarray):
        batch_size = input.shape[0]
        assert input.shape[1] == self.input_size
        assert delta.shape == (batch_size, self.output_size)

        input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)

        self._gradient += input.T @ delta

    def backward_delta(self, input, delta):
        batch_size = input.shape[0]
        assert input.shape[1] == self.input_size
        assert delta.shape == (batch_size, self.output_size)

        X = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)

        return (delta @ self._parameters.T)[:, :-1]



