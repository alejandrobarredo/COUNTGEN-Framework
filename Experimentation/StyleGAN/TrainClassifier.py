import os
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as split
from sklearn.utils import shuffle

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

os.environ["CUDA_VISIBLE_DEVICES"] = ""

dataset_path = './datasets/classifier_data/'
class_names = os.listdir(dataset_path)
# Load data (Boots, Sandals, Shoes and Slippers)
x = []
y = []

for class_count, class_name in enumerate(class_names):
    files = os.listdir(dataset_path + class_name)
    for file in tqdm(files):
        if file[-3:] == 'jpg' or file[-3:] == 'png':
            img = Image.open(dataset_path + class_name + '/' + file)
            img = img.resize((256, 256), Image.ANTIALIAS)
            img = np.array(img) / 255.0
            x.append(img)
            y.append(class_count)
        elif file[-3:] == 'npy':
            img = np.load(dataset_path + class_name + '/' + file)
            img = img / 255.0
            x.append(img)
            y.append(class_count)

x = np.stack(x, axis=0)
y = np.array(y)
x_train, x_test, y_train, y_test = split(x, y, test_size=0.1, random_state=1)

with open('datasets/classifier_data/classifier_dataset.pkl', 'wb') as f:
    pkl.dump((x_train, x_test, y_train, y_test), f, pkl.HIGHEST_PROTOCOL)
print()
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(16,
                 kernel_size=3,
                 activation='relu',
                 input_shape=(256, 256, 3)))
model.add(Dropout(0.9))
model.add(Conv2D(4,
                 kernel_size=3,
                 activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=16,
          epochs=50)

model.save('./models/classifier_model.h5')

print()