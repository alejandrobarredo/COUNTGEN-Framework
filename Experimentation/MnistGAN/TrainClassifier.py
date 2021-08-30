import os, time, itertools, imageio, pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split as split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import pickle as pkl
import copy as cp

# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)


new_x_test = cp.deepcopy(x_test)
new_y_test = cp.deepcopy(list(y_test))
# generate class of unfinished digits
idx = np.random.choice(range(10000), 1000)
blob_side = 10
fig, ax = plt.subplots(1, 5)
for count, i in enumerate(idx):
    img = x_test[i, :, :, :].reshape(28, 28)
    x_i = np.random.randint(0, 28 - blob_side)
    y_i = np.random.randint(0, 28 - blob_side)
    img[x_i:x_i + blob_side, y_i:y_i + blob_side] = 0

    new_x_test = np.concatenate([new_x_test, img.reshape(1, 28, 28, 1)])
    new_y_test.append(10)

    if count == 5:
        plt.show()
    ax[count].imshow(img)
    ax[count].set_xticks([])
    ax[count].set_yticks([])
    # plt.pause(0.1)
    # print()

new_y_test = to_categorical(new_y_test)
x_train, x_test, y_train, y_test = split(new_x_test, new_y_test, test_size=0.1,
                                         random_state=1)


with open('datasets/mnist_test_unfinished.pkl', 'wb') as f:
    pkl.dump((x_train, x_test, y_train, y_test), f, pkl.HIGHEST_PROTOCOL)
model = Sequential()
model.add(Conv2D(32, (3, 3),
                 activation='relu',
                 kernel_initializer='he_uniform',
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(11, activation='softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=16,
          epochs=30,
          verbose=1)

model.save('./Models/classifier_model_mnist_unfinished.h5')

print()