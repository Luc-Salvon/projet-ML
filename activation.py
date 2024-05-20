import math
import numpy as np
from classes_abstraites import Module



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

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        return (self.forward(input) * (1 - self.forward(input))) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass



class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp_scores = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def backward_delta(self, input, delta):
        s = self.forward(input)
        n, d = input.shape

        """
        # Diagonal elements
        diag_indices = np.arange(d)
        jacobian_matrices[:, diag_indices, diag_indices] = s * (1 - s)
        
        # Off-diagonal elements
        off_diag_indices = np.arange(d)
        jacobian_matrices[:, off_diag_indices[:, None], off_diag_indices] = -s[:, :, None] * s[:, None, :]
        
        return jacobian_matrices
        """
        return delta * s * (1-s)

    def update_parameters(self, gradient_step=1e-3):
        pass



class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def backward_delta(self, input, delta):
        return delta * (input > 0)

    def update_parameters(self, gradient_step=1e-3):
        pass
