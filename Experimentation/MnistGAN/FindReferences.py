from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

discriminator = load_model('./Models/discriminator.h5')
generator = load_model('./Models/generator.h5')
classifier = load_model('./Models/classifier_model.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
f, ax = plt.subplots(1, 1)
preds = classifier.predict(x_test.reshape(-1, 28, 28, 1)/255)
low_fours = []
low_four_idx = []
low_eights = []
low_eight_idx = []
for i in range(len(y_test)):
    if y_test[i] == 4:
        #ax.imshow(x_test[i])
        low_fours.append(preds[i][8])
        low_four_idx.append(i)
    if y_test[i] == 8:
        #ax.imshow(x_test[i])
        low_eights.append(preds[i][4])
        low_eight_idx.append(i)
    #plt.pause(0.1)

worst_four = x_test[low_four_idx[np.argmax(low_fours)]]
worst_four_pred = np.squeeze(classifier.predict(x_test[low_four_idx[np.argmax(
    low_fours)]].reshape(1, 28, 28, 1)/255))
worst_eight = x_test[low_eight_idx[np.argmax(low_eights)]]
worst_eight_pred = np.squeeze(classifier.predict(x_test[low_eight_idx[
    np.argmax(
    low_eights)]].reshape(1, 28, 28, 1)/255))



# f, ax = plt.subplots(1, 2)
# ax[0].imshow(worst_four)
# ax[1].imshow(worst_eight)
# plt.show()

with open('./Counterfactuals/References.pkl', 'wb') as f:
    references = pkl.dump([worst_four, worst_four_pred, worst_eight,
                           worst_eight_pred], f)