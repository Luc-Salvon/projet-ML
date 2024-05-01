import math
import numpy as np
from classes_abstraites import Module
from numpy.lib.stride_tricks import sliding_window_view



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




### Convolution

class Flatten(Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        out_length = (length - self.k_size) // self.stride + 1

        X_view = sliding_window_view(X, (1, self.k_size, 1))[::1, :: self.stride, ::1].reshape(batch_size, out_length, chan_in, self.k_size)

        self.output = np.max(X_view, axis=-1)
        return self.output


    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        input_view = sliding_window_view(input, (1, self.k_size, 1))[::1, :: self.stride, ::1].reshape(batch_size, out_length, chan_in, self.k_size)

        max_indices = np.argmax(input_view, axis=-1)

        batch_indices = np.arange(batch_size)[:, None, None]
        out_indices = np.arange(out_length)[None, :, None]
        chan_indices = np.arange(chan_in)[None, None, :]

        self.d_out = np.zeros_like(input)
        self.d_out[batch_indices, out_indices * self.stride + max_indices, chan_indices] += delta[batch_indices, max_indices, chan_indices]

        """
        for b in range(batch_size):
            for i in range(out_length):
                for c in range(chan_in):
                    max_index = max_indices[b, i, c]
                    self.d_out[b, i * self.stride + max_index, c] += delta[b, i, c]
        """

        return self.d_out





class Conv1D(Module):

    def __init__(self, k_size: int, chan_in: int, chan_out: int, stride: int = 1):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride

        self._parameters = np.random.uniform(0.0, 1.0, (self.k_size, self.chan_in, self.chan_out))
        self._gradient = np.zeros_like(self._parameters)


    def zero_grad(self):
        self._gradient = np.zeros_like(self._parameters)

      

    def update_parameters(self, learning_rate):
        self._parameters -= learning_rate * self._gradient
       