import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline
#%pylab inline
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
print(pd.__version__)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

# let's see what compute devices we have available, hopefully a GPU
sess = tf.Session()
devices = sess.list_devices()
print("We expect to fond a GPU in the list below...")
for d in devices:
    print(d.name)

# a small sanity check, does tf seem to work ok?
hello = tf.constant('Hello TF!')
print(sess.run(hello))

from tensorflow import keras
print(keras.__version__)

# Now, get to the data
print("Getting to the data...")
df = pd.read_csv('../data/insurance-customers-1500.csv', sep=';')

print("Data read")
df.head()
df.describe()

print("Ok...")

# same split of input and output columns as before

y = df['group']
df.drop('group', axis='columns', inplace=True)
X = df.as_matrix()

from sklearn.model_selection import train_test_split

# using stratify
# we get a balanced number of samples per category (important!)

X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# we have (almost) the same number of samples per categoery
# in the training...
np.unique(y_train, return_counts=True)

# ... and test dataset
np.unique(y_test, return_counts=True)

# ignore this, it is just technical code to plot decision boundaries
# Adapted from:
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
# http://jponttuset.cat/xkcd-deep-learning/

from matplotlib.colors import ListedColormap

cmap_print = ListedColormap(['#AA8888', '#004000', '#FFFFDD'])
cmap_bold = ListedColormap(['#AA4444', '#006000', '#EEEE44'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#FFFFDD'])
font_size = 25
title_font_size = 40


def meshGrid(x_data, y_data, xlim=None, ylim=None):
    h = 1  # step size in the mesh
    if xlim == None:
        xlim = x_data.min(), x_data.max()
    if ylim == None:
        ylim = y_data.min(), y_data.max()

    x_min, x_max = xlim
    y_min, y_max = ylim
    xx, yy = np.meshgrid(np.arange(x_min - 1, x_max + 1, h),
                         np.arange(y_min - 1, y_max + 1, h))
    return xx, yy, xlim, ylim


def plot_prediction(clf, x_data, y_data, x_label, y_label, ground_truth, title="",
                    mesh=True, fixed=None, fname=None,
                    size=(8, 5),
                    print=False, xlim=(16, 90), ylim=(70, 170)):
    xx, yy, xlim, ylim = meshGrid(x_data, y_data, xlim, ylim)
    fig, ax = plt.subplots(figsize=size)

    if clf and mesh:
        grid_X = np.array(np.c_[yy.ravel(), xx.ravel()])
        if fixed:
            fill_values = np.full((len(grid_X), 1), fixed)
            grid_X = np.append(grid_X, fill_values, axis=1)
        Z = clf.predict(grid_X)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if print:
        ax.scatter(x_data, y_data, c=ground_truth, cmap=cmap_print, s=200, marker='o', edgecolors='k')
    else:
        ax.scatter(x_data, y_data, c=ground_truth, cmap=cmap_bold, s=100, marker='o', edgecolors='k')

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_title(title, fontsize=title_font_size)
    if fname:
        fig.savefig('figures/' + fname)


def plot_history(history, samples=100, init_phase_samples=None,
                 size=(8, 6),
                 plot_line=False):
    epochs = history.params['epochs']

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    every_sample = int(epochs / samples)
    acc = pd.DataFrame(acc).iloc[::every_sample, :]
    val_acc = pd.DataFrame(val_acc).iloc[::every_sample, :]
    loss = pd.DataFrame(loss).iloc[::every_sample, :]
    val_loss = pd.DataFrame(val_loss).iloc[::every_sample, :]

    if init_phase_samples:
        acc = acc.loc[init_phase_samples:]
        val_acc = val_acc.loc[init_phase_samples:]
        loss = loss.loc[init_phase_samples:]
        val_loss = val_loss.loc[init_phase_samples:]

    fig, ax = plt.subplots(nrows=2, figsize=size)

    ax[0].plot(acc, 'bo', label='Training acc')
    ax[0].plot(val_acc, 'b', label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()

    if plot_line:
        x, y, _ = linear_regression(acc)
        ax[0].plot(x, y, 'bo', color='red')
        x, y, _ = linear_regression(val_acc)
        ax[0].plot(x, y, 'b', color='red')

    ax[1].plot(loss, 'bo', label='Training loss')
    ax[1].plot(val_loss, 'b', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()

    if plot_line:
        x, y, _ = linear_regression(loss)
        ax[1].plot(x, y, 'bo', color='red')
        x, y, _ = linear_regression(val_loss)
        ax[1].plot(x, y, 'b', color='red')


from sklearn import linear_model


def linear_regression(data):
    x = np.array(data.index).reshape(-1, 1)
    y = data.values.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    return x, y_pred, regr.coef_

plot_prediction(None, X_train[:, 1], X_train[:, 0],
               'Age', 'Max Speed', y_train, mesh=False,
                title="Train Data")

plot_prediction(None, X_test[:, 1], X_test[:, 0],
               'Age', 'Max Speed', y_test, mesh=False,
                title="Test Data")

X_train_2_dim = X_train[:, :2]
X_test_2_dim = X_test[:, :2]

num_categories = 3

model = keras.Sequential()

from tensorflow.keras.layers import Dense

model.add(Dense(500, name='hidden1', activation='tanh', input_dim=2))
model.add(Dense(500, name='hidden2', activation='tanh'))

model.add(Dense(num_categories, name='softmax', activation='softmax'))

model.summary()

plot_prediction(model, X_train_2_dim[:, 1], X_train_2_dim[:, 0],
               'Age', 'Max Speed', y_train,
                title="Untrained")

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# only if you are running this locally

# https://keras.io/callbacks/#tensorboard
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./tf_log')
# To start tensorboard
# tensorboard --logdir=./tf_log
# open http://localhost:6006

BATCH_SIZE=1000
EPOCHS = 5000

# only if you are running this locally
# !rm -rf ./tf_log
# %time model.fit(X_train_2_dim, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[tb_callback])

# %time history = model.fit(X_train_2_dim, y_train, \
history = model.fit(X_train_2_dim, y_train, \
                          epochs=EPOCHS, batch_size=BATCH_SIZE, \
                          validation_split=0.2, verbose=0)

train_loss, train_accuracy = \
  model.evaluate(X_train_2_dim, y_train, batch_size=BATCH_SIZE)
train_accuracy

test_loss, test_accuracy = \
  model.evaluate(X_test_2_dim, y_test, batch_size=BATCH_SIZE)
test_accuracy

plot_history(history)

plot_prediction(model, X_train_2_dim[:, 1], X_train_2_dim[:, 0],
               'Age', 'Max Speed', y_train,
                title="Train Data")

plot_prediction(model, X_test_2_dim[:, 1], X_test_2_dim[:, 0],
               'Age', 'Max Speed', y_test,
                title="Test Data")
