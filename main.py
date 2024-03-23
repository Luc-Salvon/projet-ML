from matplotlib import pyplot as plt

import mltools

from modele import Sequentiel, sgd
from losses import *
from modules import *


def test_partie1():
    net = Sequentiel([Linear(2, 1)])

    data = mltools.gen_arti(nbex=1000, data_type=0)
    X_train, X_test, Y_train, Y_test = data[0][:800], data[0][800:], data[1][:800], data[1][800:]

    evolution_loss = sgd(net, (X_train, Y_train), loss=MSELoss())

    # Plot de la loss
    plt.plot(evolution_loss)
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.show()

    # Frontiere de decision
    mltools.plot_frontiere(X_test, net.forward)
    mltools.plot_data(X_test, Y_test)
    plt.title("Frontiere de decision")
    plt.show()


def test_partie2():
    net = Sequentiel([Linear(2, 3), TanH(), Linear(3, 1), Sigmoide()])

    data = mltools.gen_arti(nbex=1000, data_type=1)
    X_train, X_test, Y_train, Y_test = data[0][:800], data[0][800:], data[1][:800], data[1][800:]
    Y_train[Y_train == -1] = 0
    Y_test[Y_test == -1] = 0

    evolution_loss = sgd(net, (X_train, Y_train), loss=MSELoss(), nb_epochs=1000)

    # Plot de la loss
    plt.plot(evolution_loss)
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.show()

    # Frontiere de decision
    mltools.plot_frontiere(X_test, lambda x: net.forward(x) - .5)
    mltools.plot_data(X_test, Y_test)
    plt.title("Frontiere de decision")
    plt.show()

    yhat = net.forward(X_test)
    print("Accuracy:", np.mean(np.round(yhat) == Y_test))



def tes_partie4():
    pass

if __name__ == "__main__":
    # test_partie1()
    test_partie2()
