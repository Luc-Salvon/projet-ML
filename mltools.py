import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from modele import Sequentiel,sgd
from modules import Linear
from losses import MSELoss


def plot_data(data, labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], [".", "+", "*", "o", "x", "^"]
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], marker="x")
        return
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[i], marker=marks[i])


def plot_frontiere(data, f, step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), colors=('gray', 'blue'), levels=[-1, 0, 1])


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), np.max(data[:, 1]), np.min(data[:, 1])
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step), np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type == 0:
        # melange de 2 gaussiennes
        xpos = np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 2)
        xneg = np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 2)
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack((np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        xneg = np.vstack((np.random.multivariate_normal([-centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))

    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2 * nbex), (nbex, 2))
        y = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        y = 2 * (y % 2) - 1
    # un peu de bruit
    data[:, 0] += np.random.normal(0, epsilon, nbex)
    data[:, 1] += np.random.normal(0, epsilon, nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
    return data, y.reshape(-1, 1)



def bruitage_données(X,epsilon=0.1):
    '''
    X : données
    epsilon : Probabilité du bruit
    '''
    X += np.random.normal(0, epsilon, size=X.shape)
    return X




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def grid_search(net, param_grid, X, y, cv=3):
    """
    Perform a grid search over the given parameter grid.
    
    Parameters:
    model: the neural network
    param_grid: dictionary where keys are parameter names and values are lists of parameter settings to try
    X, y: data
    cv: number of cross-validation folds
    
    Returns:
    best_params: the parameter set that gave the best mean cross-validation score
    best_score: the best mean cross-validation score
    """

    def cross_val_score(model, X, y, cv):
        scores = []
        X_split, y_split = np.array_split(X, cv), np.array_split(y, cv)
        for i in range(cv):
            X_train = np.concatenate(X_split[:i] + X_split[i+1:])
            y_train = np.concatenate(y_split[:i] + y_split[i+1:])
            X_val = X_split[i]
            y_val = y_split[i]
            
            sgd(net, (X, y), loss=MSELoss())
            y_pred = net.forward(X_val)
            scores.append(accuracy_score(y_val, y_pred))
        return np.mean(scores)

    best_params = None
    best_score = -np.inf
    
    # Generate all combinations of parameters
    import itertools
    keys, values = zip(*param_grid.items())
    for param_combination in itertools.product(*values):
        params = dict(zip(keys, param_combination))
        net.set_params(**params)
        score = cross_val_score(net, X, y, cv)
        print(f"Params: {params}, Score: {score}")
        
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


"""
# Example 
net = Sequentiel([Linear(2, 1)])
data = gen_arti(nbex=1000, data_type=0)
X, y = data[0], data[1]
    
param_grid = {
    'learning_rate': [1e-5, 1e-3, 1e-2],
    'batch_size': [10, 32, 64]
}
best_params, best_score = grid_search(net, param_grid, X, y, cv=3)

print("Best Parameters:", best_params)
print("Best Score:", best_score)
"""
