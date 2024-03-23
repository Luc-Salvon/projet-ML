import numpy as np

from classes_abstraites import Loss


class MSELoss(Loss):
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return np.linalg.norm(y - yhat, axis=1)**2

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return -2 * (y - yhat)


class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        assert y.shape == yhat.shape

        return -np.sum(y * np.log(yhat), axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape

        return yhat - y


class LogSoftmaxCrossEntropy(Loss):

    def forward(self, y, yhat):
        return CrossEntropyLoss().forward(y, yhat) + np.log(np.sum(np.exp(yhat), axis=1))

    def backward(self, y, yhat):
        return CrossEntropyLoss().backward(y, yhat) + np.exp(yhat) / np.sum(np.exp(yhat), axis=1)[:, None]