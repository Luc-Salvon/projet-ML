import numpy as np

from classes_abstraites import Loss


class MSELoss(Loss):
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return np.linalg.norm(y - yhat, axis=1)**2

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return -2 * (y - yhat)


class CrossEntropyLoss(Loss):
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return 1 - np.sum(y*yhat,axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray): # pas sur
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return yhat - y


class LogSoftmaxCrossEntropy(Loss):

    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return np.log(np.exp(yhat).sum(axis=1)) - (y * yhat).sum(axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        return np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape((-1, 1)) - y


class BCELoss(Loss):
    # Binary Cross Entropy loss

    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        # Clip yhat to prevent log(0)
        return -np.mean(y * np.log(np.clip(yhat, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - yhat, 1e-10, 1)))

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape, ValueError(f"y has shape {y.shape} and yhat has shape {yhat.shape}")

        # Clip yhat to prevent division by zero
        return -(y / np.clip(yhat, 1e-10, 1) - (1 - y) / np.clip(1 - yhat, 1e-10, 1))