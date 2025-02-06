import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def svm_playground(dataset, kernel='rbf', C=1, gamma='scale', degree=3, noise=0.2, n_samples=1000, random_state=307):
    # Load dataset
    if dataset == 'blobs':
        X, y = datasets.make_blobs(n_samples=n_samples, centers=[[.75,.75],[-.75,-.75]], cluster_std=noise*3, random_state=random_state)
    elif dataset == 'moons':
        X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset == 'circles':
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=random_state)
    else:
        return "Invalid dataset name. Choose from 'blobs', 'moons', or 'circles'."

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM classification
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    clf.fit(X_train, y_train)
    
    # Calculate accuracy
    train_accuracy = np.round(accuracy_score(y_train, clf.predict(X_train)),4)
    test_accuracy = np.round(accuracy_score(y_test, clf.predict(X_test)),4)

    
    # Plot decision boundary
    plot_svm_decision_boundary(clf, X, y, dataset, train_accuracy, test_accuracy, scaler)


def plot_svm_decision_boundary(clf, X, y, title, train_accuracy, test_accuracy, scaler):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # step size in the mesh
    adjust = .25
    x_min, x_max = X[:, 0].min() - adjust, X[:, 0].max() + adjust
    y_min, y_max = X[:, 1].min() - adjust, X[:, 1].max() + adjust
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    meshgrid_array = np.c_[xx.ravel(), yy.ravel()]
    meshgrid_array_scaled = scaler.transform(meshgrid_array)

    Z = clf.predict(meshgrid_array_scaled)
    Z = Z.reshape(xx.shape)

    params = clf.get_params()
    kernel = params['kernel']
    C = params['C']
    gamma = params['gamma']
    #degree = params['degree']

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2, width_ratios=[2,1])
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(xx, yy, Z, cmap=cmap_light)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_title(f"SVM Decision Boundary for {title} dataset")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    textstr = '\n'.join((
        'Performance:',
        f'Train accuracy: {train_accuracy}',
        f'Test accuracy: {test_accuracy}',
        '',
        '---',
        '',
        'Model Parameters:',
        f'Kernel: {kernel}',
        f'C: {C}',
        f'gamma: {gamma}',
        # f'degree: {degree}',
        '',
        #'---',
        #'',
        #'"degree" ignored if ',
        #'kernel is not "poly"'
    ))
    ax2.text(0.5, 0.5, textstr, transform=ax2.transAxes, fontsize=14, va='center', ha='center')
    plt.tight_layout()
    plt.show()




def gb_playground(dataset, n_estimators=100, learning_rate=.1, max_depth=3, noise=.2, n_samples=1000, random_state=307):
    # Load dataset
    if dataset == 'blobs':
        X, y = datasets.make_blobs(n_samples=n_samples, centers=[[.75,.75],[-.75,-.75]], cluster_std=noise*3, random_state=random_state)
    elif dataset == 'moons':
        X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset == 'circles':
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    else:
        return "Invalid dataset name. Choose from 'blobs', 'moons', or 'circles'."

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

    # Gradient Boosting classification
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=6)
    clf.fit(X_train, y_train)

    # Calculate accuracy
    train_accuracy = np.round(accuracy_score(y_train, clf.predict(X_train)),4)
    test_accuracy = np.round(accuracy_score(y_test, clf.predict(X_test)),4)

    # Plot decision boundary
    plot_gb_decision_boundary(clf, X, y, dataset, train_accuracy, test_accuracy)  

    #return train_accuracy, test_accuracy



def plot_gb_decision_boundary(clf, X, y, title, train_accuracy, test_accuracy):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # step size in the mesh
    adjust = .25
    x_min, x_max = X[:, 0].min() - adjust, X[:, 0].max() + adjust
    y_min, y_max = X[:, 1].min() - adjust, X[:, 1].max() + adjust
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    params = clf.get_params()
    max_depth = params['max_depth']
    learning_rate = params['learning_rate']
    n_estimators = params['n_estimators']

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2, width_ratios=[2,1])
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(xx, yy, Z, cmap=cmap_light)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_title(f"BG Decision Boundary for {title} dataset")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    textstr = '\n'.join((
        'Performance:',
        f'Train accuracy: {train_accuracy}',
        f'Test accuracy: {test_accuracy}',
        '',
        '---',
        '',
        'Model Parameters:',
        f'max_depth: {max_depth}',
        f'learning_rate: {learning_rate}',
        f'n_estimators: {n_estimators}'
    ))
    ax2.text(0.5, 0.5, textstr, transform=ax2.transAxes, fontsize=14, va='center', ha='center')
    plt.tight_layout()
    plt.show()
