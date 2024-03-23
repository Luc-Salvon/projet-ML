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


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return (1 - (np.tanh(input)**2)) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def sigma(self, X):
        return 1 / (1 + np.exp(-X))

    def forward(self, X):
        return self.sigma(X)

    def backward_delta(self, input, delta):
        return (self.sigma(input) * (1 - self.sigma(input))) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass



class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def backward_delta(self, input, delta):
        s = self.forward(input)
        n, d = input.shape

        # Diagonal elements
        diag_indices = np.arange(d)
        jacobian_matrices[:, diag_indices, diag_indices] = s * (1 - s)
        
        # Off-diagonal elements
        off_diag_indices = np.arange(d)
        jacobian_matrices[:, off_diag_indices[:, None], off_diag_indices] = -s[:, :, None] * s[:, None, :]
        
        return jacobian_matrices

    def update_parameters(self, gradient_step=1e-3):
        pass

