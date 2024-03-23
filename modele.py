from typing import Iterable

import classes_abstraites


class Modele:
    def __init__(self, modules: Iterable[classes_abstraites.Module], loss: classes_abstraites.Loss):
        self.modules = modules  # Liste des modules utilisés dans le modèle
        self.loss = loss  # Fonction de loss utilisée
        self.loss_evolution = []  # Liste de la loss à chaque epoch

    def fit(self, X, Y, nb_epochs=50, eps=10e-5, batch_size=5):
        N = X.shape[0]  # Nombre d'exemples total

        for i_epoch in range(nb_epochs):  # On itère sur les epochs
            self.loss_evolution.append(self.loss.forward(Y, self.predict(X)).mean())

            # Si la loss n'évolue pas significativement on s'arrête
            if len(self.loss_evolution) > 1 and abs(self.loss_evolution[-1] - self.loss_evolution[-2]) < eps:
                break

            for i_batch in range(0, N, batch_size):  # On itère sur les batchs
                X_batch = X[i_batch:min(i_batch + batch_size, N)]
                Y_batch = Y[i_batch:min(i_batch + batch_size, N)]

                Yhat = self.predict(X_batch)

                delta = self.loss.backward(Y_batch, Yhat)

                # Backpropagation
                for i, module in enumerate(self.modules[::-1]):
                    module.zero_grad()
                    module.backward_update_gradient(self.inputs[-(i+2)], delta)
                    module.update_parameters()
                    delta = module.backward_delta(self.inputs[-(i+2)], delta)

    def predict(self, X):
        self.inputs = [X]

        for module in self.modules:
            self.inputs.append(module.forward(self.inputs[-1]))

        return self.inputs[-1]
