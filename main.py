from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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



def test_partie4():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X /= 255.0 # Normalisation des données

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    onehot_train = np.zeros((y_train.size,10))
    onehot_train[np.arange(y_train.size),y_train]=1

    onehot_test = np.zeros((y_test.size,10))
    onehot_test[np.arange(y_test.size),y_test]=1

    net = Sequentiel([Linear(X.shape[1], 10)])
    
    evolution_loss = sgd(net, (X_train, onehot_train), loss=LogSoftmaxCrossEntropy(), batch_size=32, nb_epochs=100, eps=1e-2)

    # Plot de la loss
    plt.plot(evolution_loss)
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.show()

    # Évaluation du modèle sur l'ensemble de test
    pred = np.argmax(net.forward(X_test), axis=1)
    pred_one_hot = np.zeros_like(onehot_test)
    pred_one_hot[np.arange(onehot_test.shape[0]), pred] = 1

    accuracy = np.mean(pred_one_hot == onehot_test)
    print("Accuracy sur l'ensemble de test:", accuracy)

if __name__ == "__main__":
    # test_partie1()
    # test_partie2()
    test_partie4()
