import numpy as np

from classes_abstraites import Loss


class MSELoss(Loss):
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return np.linalg.norm(y - yhat, axis=1)**2

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return -2 * (y - yhat)
