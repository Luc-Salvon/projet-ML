from matplotlib import pyplot as plt

import mltools

from modele import Modele
from losses import *
from modules import *


if __name__ == "__main__":
    modele = Modele([Linear(2, 1)], MSELoss())

    data = mltools.gen_arti(nbex=1000, data_type=0)
    X_train, X_test, Y_train, Y_test = data[0][:800], data[0][800:], data[1][:800], data[1][800:]

    modele.fit(X_train, Y_train)

    # Plot de la loss
    plt.plot(modele.loss_evolution)
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.show()

    # Frontiere de decision
    mltools.plot_frontiere(X_test, modele.predict)
    mltools.plot_data(X_test, Y_test)
    plt.title("Frontiere de decision")
    plt.show()
