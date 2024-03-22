import numpy as np

from classes_abstraites import Module


class Linear(Module):
    def __init__(self, input: int, output: int):
        super().__init__()

        self.input_size = input
        self.output_size = output

        self._parameters = np.random.random((self.input_size, self.output_size))
        self._gradient = np.zeros((self.input_size, self.output_size))

    def zero_grad(self):
        self._gradient.fill(0)

    def forward(self, X: np.ndarray):
        assert X.shape[1] == self.input_size

        return X @ self._parameters

    def backward_update_gradient(self, input, delta):
        pass  # TODO

    def backward_delta(self, input, delta):
        pass  # TODO