from typing import Iterable

import numpy as np

import classes_abstraites


class Sequentiel:
    def __init__(self, modules: Iterable[classes_abstraites.Module]):
        self.modules = modules  # Liste des modules utilisés dans le modèle

    def forward(self, X):
        self.inputs = [X]

        for module in self.modules:
            self.inputs.append(module.forward(self.inputs[-1]))

        return self.inputs[-1]

    def backward(self, input, delta):
        # Backprop -> On calcule le gradient de la fin vers le début
        for i, module in enumerate(self.modules[::-1]):
            module.backward_update_gradient(self.inputs[-(i + 2)], delta)
            delta = module.backward_delta(self.inputs[-(i + 2)], delta)


class Optim:
    def __init__(self, net: Sequentiel, loss: classes_abstraites.Loss, eps: float):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_X, batch_Y):
        Yhat = self.net.forward(batch_X)

        delta = self.loss.backward(batch_Y, Yhat)

        # Calcul du gradient
        self.net.backward(batch_X, delta)

        # Update des paramètres
        for module in self.net.modules:
            module.update_parameters(gradient_step=self.eps)
            module.zero_grad()


def sgd(net: Sequentiel, data: tuple[np.ndarray], loss: classes_abstraites.Loss, batch_size: int = 5, nb_epochs: int = 50, eps: float = 10e-5, step: float = 1e-3):
    X, Y = data[:]
    N = X.shape[0]  # Nombre d'exemples total

    optim = Optim(net, loss, step)

    evolution_loss = []

    for i_epoch in range(nb_epochs):  # On itère sur les epochs
        evolution_loss.append(loss.forward(Y, net.forward(X)).mean())

        # Si la loss n'évolue pas significativement on s'arrête
        if len(evolution_loss) > 1 and abs(evolution_loss[-1] - evolution_loss[-2]) < eps:
            break

        for i_batch in range(0, N, batch_size):  # On itère sur les batchs
            batch_X = X[i_batch:min(i_batch + batch_size, N)]
            batch_y = Y[i_batch:min(i_batch + batch_size, N)]

            optim.step(batch_X, batch_y)

    return evolution_loss



class AutoEncoder():
    def __init__(self,activation):
        self.encoder = Sequentiel([Linear(256, 100), activation, Linear(100, 10), activation])
    
        self.decoder = Sequentiel([Linear(10, 100), activation, Linear(100, 256), Sigmoid()])

    def forward(self,data):
        return self.decoder(self.encoder(data))
    
    def backward(self, input, delta):
        # TODO
        pass

