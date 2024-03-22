import numpy as np

from classes_abstraites import Loss


class MSELoss(Loss):
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        batch_size, d = y.shape
        assert yhat.shape == (batch_size, d)

        return np.linalg.norm(y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        pass  # TODO
