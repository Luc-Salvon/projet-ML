from typing import Iterable

import numpy as np

import classes_abstraites


class Modele:
    def __init__(self, modules: Iterable[classes_abstraites.Module], loss: classes_abstraites.Loss):
        self.modules = modules
        self.loss = loss
        self.loss_evolution = []

    def fit(self, X, Y, nb_epochs=50, eps=10e-4, batch_size=5):
        N = X.shape[0]

        for i_epoch in range(nb_epochs):
            if len(self.loss_evolution) > 1 and abs(self.loss_evolution[-1] - self.loss_evolution[-2]) < eps:
                break

            self.loss_evolution.append(self.loss.forward(Y, self.predict(X)).mean())

            for i_batch in range(0, N, batch_size):
                X_batch = X[i_batch:min(i_batch + batch_size, N)]
                Y_batch = Y[i_batch:min(i_batch + batch_size, N)]

                Yhat = self.predict(X_batch)

                loss = self.loss.forward(Y_batch, Yhat).mean()

                delta = self.loss.backward(Y_batch, Yhat)

                for module in self.modules:
                    module.zero_grad()
                    module.backward_update_gradient(X_batch, delta)
                    module.update_parameters()
                    delta = module.backward_delta(X_batch, delta)

    def predict(self, X):
        for module in self.modules:
            X = module.forward(X)

        return X
