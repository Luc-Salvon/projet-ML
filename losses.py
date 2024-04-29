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
    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return 1 - np.sum(y*yhat,axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray): # pas sur
        assert y.shape == yhat.shape

        return yhat - y


class LogSoftmaxCrossEntropy(Loss):

    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return CrossEntropyLoss().forward(y, yhat) + np.log(np.sum(np.exp(yhat), axis=1))

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return CrossEntropyLoss().backward(y, yhat) + np.exp(yhat) / np.sum(np.exp(yhat), axis=1)[:, None]


class BCELoss(Loss):
    # Binary Cross Entropy loss

    def forward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        #return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y: np.ndarray, yhat: np.ndarray):
        assert y.shape == yhat.shape

        return - (y / yhat - (1 - y) / (1 - yhat))