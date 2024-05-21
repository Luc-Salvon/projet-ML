import math
import numpy as np
from classes_abstraites import Module
from numpy.lib.stride_tricks import sliding_window_view
from typing import Literal



class Linear(Module):
    def __init__(self, input: int, output: int):
        super().__init__()

        self.input_size = input
        self.output_size = output

        self._parameters = np.random.uniform(-0.1, 0.1, (self.input_size + 1, self.output_size))
        self._gradient = np.zeros((self.input_size + 1, self.output_size))

    def zero_grad(self):
        self._gradient.fill(0)

    def forward(self, X: np.ndarray):
        assert X.shape[1] == self.input_size, ValueError(f"shape doesn't match: data is {X.shape[1]} but should have {self.input_size}")

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
    
    def update_parameters(self, gradient_step=1e-3):
        pass


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape

        dout = (length - self.k_size) // self.stride + 1

        indices = np.arange(0, dout * self.stride, self.stride).reshape(-1, 1) + np.arange(self.k_size)  # shape : dout x k_size
        segments = X[:, indices, :]  # shape : batch_size x dout x k_size x chan_in
        output = np.max(segments, axis=2)  # shape : batch_size x dout x chan_in
        
        return output
    

    def backward_delta(self, input, delta): # marche

        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        input_view = sliding_window_view(input, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        input_view = input_view.reshape(batch_size, out_length, chan_in, self.k_size)

        max_indices = np.argmax(input_view, axis=-1)

        # Create indices for batch and channel dimensions
        batch_indices, out_indices, chan_indices = np.meshgrid(
            np.arange(batch_size),
            np.arange(out_length),
            np.arange(chan_in),
            indexing="ij",
        )

        # Update d_out using advanced indexing
        delta_input = np.zeros_like(input)
        delta_input[batch_indices, out_indices * self.stride + max_indices, chan_indices] += delta[batch_indices, max_indices, chan_indices]

        return delta_input
    

    def backward_delta_2(self, input, delta): #long
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1
        
        # Initialize delta_input
        delta_input = np.zeros_like(input)
        
        for b in range(batch_size):
            for c in range(chan_in):
                for i in range(out_length):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.k_size
                    max_idx = np.argmax(input[b, start_idx:end_idx, c])
                    
                    # Update delta_input
                    delta_input[b, start_idx + max_idx, c] += delta[b, i, c]

        return delta_input


    
    def backward_delta_3(self, input, delta): # marche pas
        batch_size, length, chan_in = input.shape
        dout = (length - self.k_size) // self.stride + 1

        indices = np.arange(0, dout * self.stride, self.stride).reshape(-1, 1) + np.arange(self.k_size)  # shape : dout x k_size
        segments = input[:, indices, :]  # shape : batch_size x dout x k_size x chan_in
        max_indices = np.argmax(segments, axis=2) + indices[:, 0].reshape(1, -1, 1)  # shape : batch_size x dout x chan_in


        batch_indices = np.arange(batch_size)[:, None, None]
        dout_indices = np.arange(dout)[None, :, None]
        chan_indices = np.arange(chan_in)[None, None, :]

        delta_input = np.zeros_like(input)
        delta_input[batch_indices, dout_indices * self.stride + max_indices, chan_indices] += delta[batch_indices, max_indices, chan_indices]
        return delta_input
    
    def update_parameters(self, gradient_step=1e-3):
        pass




class Conv1D(Module):

    def __init__(self, k_size: int, chan_in: int, chan_out: int, stride: int = 1):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride

        scale_factor = 0.1 
        self._parameters = np.random.uniform(-1.0 * scale_factor, 1.0 * scale_factor, (self.k_size, self.chan_in, self.chan_out))

        self._gradient = np.zeros_like(self._parameters)


    def zero_grad(self):
        self._gradient = np.zeros_like(self._parameters)

       
    def forward(self, X):
        batch_size, length, chan_in = X.shape
        assert chan_in == self.chan_in, ValueError(f"number of channels doesn't match: data has {chan_in} but should have {self.chan_in}")
        
        dout = (length - self.k_size) // self.stride + 1

        indices = np.arange(0, dout * self.stride, self.stride).reshape(-1, 1) + np.arange(self.k_size) # shape : dout x k_size

        segments = X[:, indices, :] # shape : batch_size x dout x k_size x chan_in
        output = np.tensordot(segments, self._parameters, axes=((2, 3), (0, 1)))

        return output


    def backward_update_gradient(self, input, delta): # pas sur du tout
        batch_size, length, chan_in = input.shape
        assert chan_in == self.chan_in, ValueError(f"number of channels doesn't match: data has {chan_in} but should have {self.chan_in}")

        dout = delta.shape[1]
        indices = np.arange(0, dout * self.stride, self.stride).reshape(-1, 1) + np.arange(self.k_size)
        segments = input[:, indices, :]  # shape : batch_size x dout x k_size x chan_in
        
        self._gradient += np.sum(np.tensordot(segments, delta, axes=((0, 1), (0, 1))), axis=0)


        
    def backward_delta(self, input, delta): # pas sur
        batch_size, length, chan_in = input.shape
        assert chan_in == self.chan_in, ValueError(f"number of channels doesn't match: data has {chan_in} but should have {self.chan_in}")
  
        dout = delta.shape[1]
        delta_input = np.zeros_like(input)
        
        indices = np.arange(0, dout * self.stride, self.stride).reshape(-1, 1) + np.arange(self.k_size)
        
        for i in range(self.k_size):
            delta_input[:, indices[:, i], :] += np.tensordot(delta, self._parameters[i], axes=(2, 1))
        
        return delta_input
