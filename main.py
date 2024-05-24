from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import mltools

from modele import Sequentiel, sgd, AutoEncoder, Optim
from losses import *
from modules import *
from activation import *

def load_mnist():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X /= 255.0 # Normalisation des données

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

    onehot_train = np.zeros((y_train.size,10))
    onehot_train[np.arange(y_train.size),y_train]=1

    onehot_test = np.zeros((y_test.size,10))
    onehot_test[np.arange(y_test.size),y_test]=1

    return X_train, X_test, onehot_train, onehot_test


def test_partie1():
    net = Sequentiel([Linear(2, 1)])

    data = mltools.gen_arti(nbex=1000, data_type=0)
    X_train, X_test, Y_train, Y_test = data[0][:800], data[0][800:], data[1][:800], data[1][800:]
    
    evolution_loss = sgd(net, (X_train, Y_train), (X_test, Y_test), loss=MSELoss())

    # Plot de la loss
    plt.plot(evolution_loss[0],label="train")
    plt.plot(evolution_loss[1],label='test')
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.legend()
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

    evolution_loss = sgd(net, (X_train, Y_train), (X_test, Y_test), loss=MSELoss(), nb_epochs=1000)

    # Plot de la loss
    plt.plot(evolution_loss[0],label="train")
    plt.plot(evolution_loss[1],label='test')
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.legend()
    plt.show()

    # Frontiere de decision
    mltools.plot_frontiere(X_test, lambda x: net.forward(x) - .5)
    mltools.plot_data(X_test, Y_test)
    plt.title("Frontiere de decision")
    plt.show()

    yhat = net.forward(X_test)
    print("Accuracy:", np.mean(np.round(yhat) == Y_test))



def test_partie4():
    X_train, X_test, onehot_train, onehot_test = load_mnist()

    net = Sequentiel([Linear(X_train.shape[1], 10)])
    
    evolution_loss = sgd(net, (X_train, onehot_train),(X_test, onehot_test), loss=LogSoftmaxCrossEntropy(), batch_size=32, nb_epochs=100, eps=1e-2)

    # Plot de la loss
    plt.plot(evolution_loss[0],label="train")
    plt.plot(evolution_loss[1],label='test')
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.legend()
    plt.show()

    # Évaluation du modèle sur l'ensemble de test
    pred = np.argmax(net.forward(X_test), axis=1)
    pred_one_hot = np.zeros_like(onehot_test)
    pred_one_hot[np.arange(onehot_test.shape[0]), pred] = 1

    accuracy = np.mean(pred_one_hot == onehot_test)
    print("Accuracy sur l'ensemble de test:", accuracy)




def test_encodeur_images_compressees():
    # visualiser les images reconstruites après une forte compression

    X_train, X_test, onehot_train, onehot_test = load_mnist()


    # Linear

    encoder_lin = Sequentiel([Linear(X_train.shape[1], 200), TanH(), Linear(200, 100), TanH()])
    decoder_lin = Sequentiel([Linear(100,X_train.shape[1]),Sigmoide()])
    autoencoder_lin = AutoEncoder(encoder_lin,decoder_lin)
    
    evolution_loss = sgd(autoencoder_lin, (X_train, X_train), (X_test, X_test), loss=BCELoss(), batch_size=64, nb_epochs=1000, eps=1e-8)

    reconstructed_data_lin = autoencoder_lin.forward(X_test)
    
    
    
    # Reshape the reconstructed images to their original dimensions
    num_images = X_test.shape[0]
    original_shape = (8, 8)
    reconstructed_images = reconstructed_data_lin.reshape(num_images, *original_shape)[:10]
    original_images = X_test.reshape(num_images, *original_shape)[:10]

    num_images_affichees = 5 # Show the first 5 digits
    
    # Plot original and reconstructed images 
    plt.figure(figsize=(10, 5))
    for i in range(num_images_affichees):
        plt.subplot(2, num_images_affichees, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(2, num_images_affichees, i + 1 + num_images_affichees)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Linear', fontsize=16)
    plt.show()
    

    # Convolution
    encoder_conv = Sequentiel([Conv1D(3,1,5),MaxPool1D(2,2),Flatten(),Linear(155,100),ReLU()])
    decoder_conv = Sequentiel([Linear(100,X_train.shape[1]),Sigmoide()])
    autoencoder_conv = AutoEncoder(encoder_conv,decoder_conv)

    evolution_loss = sgd(autoencoder_conv, (X_train.reshape((X_train.shape[0],X_train.shape[1],1)), X_train), (X_test.reshape((X_test.shape[0],X_test.shape[1],1)), X_test), loss=BCELoss(), batch_size=64, nb_epochs=1000, eps=1e-9)

    reconstructed_data_conv = autoencoder_conv.forward(X_test)
    
    # Reshape the reconstructed images to their original dimensions
    reconstructed_images = reconstructed_data_conv.reshape(num_images, *original_shape)[:10]
    original_images = X_test.reshape(num_images, *original_shape)[:10]


    # Plot original and reconstructed images
    plt.figure(figsize=(10, 5))
    for i in range(num_images_affichees):
        plt.subplot(2, num_images_affichees, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(2, num_images_affichees, i + 1 + num_images_affichees)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Convolution')
    plt.show()


def test_partie6():
    X_train, X_test, onehot_train, onehot_test = load_mnist()
    net = Sequentiel([Conv1D(3,1,32),MaxPool1D(2,2),Flatten(),Linear(992,100),ReLU(),Linear(100,10)])
    evolution_loss = sgd(net, (X_train, onehot_train), (X_test, onehot_test), loss=LogSoftmaxCrossEntropy(), batch_size=32, nb_epochs=10, eps=1e-10, step=1e-2)

    # Plot de la loss
    plt.plot(evolution_loss[0],label="train")
    plt.plot(evolution_loss[1],label='test')
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Loss")
    plt.title("Evolution de la loss en fonction du nombre d'epochs")
    plt.legend()
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
    # test_partie4()
    #test_encodeur_images_compressees()
    test_partie6()
