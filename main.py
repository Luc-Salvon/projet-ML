from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import mltools

from modele import Sequentiel, sgd, AutoEncoder, Optim
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



def train_autoencoder(autoencoder, data, num_epochs=10000, eps=1e-5):
    optimizer = Optim(autoencoder, loss=MSELoss(), eps=eps)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i,batch_data in enumerate(data): 
            reconstructed_data = autoencoder.forward(batch_data)
            loss = MSELoss().forward(batch_data, reconstructed_data)
            total_loss += loss.mean()
            
            # Backpropagation
            delta = MSELoss().backward(batch_data, reconstructed_data)
            autoencoder.backward(batch_data, delta)
            
            # Parameter update
            optimizer.step(batch_data, reconstructed_data, autoencoder=True)
        
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(data)}")



def test_encodeur_images_compressees():
    # visualiser les images reconstruites après une forte compression

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # y to one_hot
    onehot_train = np.zeros((y_train.size,10))
    onehot_train[np.arange(y_train.size),y_train]=1

    onehot_test = np.zeros((y_test.size,10))
    onehot_test[np.arange(y_test.size),y_test]=1


    autoencoder = AutoEncoder(input_size = X_train.shape[1],hidden_size_1=40)
        
    data_batches = np.array_split(X_train, len(X_train) // 32)
    train_autoencoder(autoencoder, data_batches)


    reconstructed_data = autoencoder.forward(X_test)
    
    # Reshape the reconstructed images to their original dimensions
    num_images = X_test.shape[0]
    original_shape = (8, 8)
    reconstructed_images = reconstructed_data.reshape(num_images, *original_shape)[:10]
    original_images = X_test.reshape(num_images, *original_shape)[:10]

    num_images = 5 # Show the first 5 digits
    
    # Plot original and reconstructed images
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_partie1()
    # test_partie2()
    # test_partie4()
    test_encodeur_images_compressees()
