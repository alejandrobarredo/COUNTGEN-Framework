import pickle as pkl
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

classifier = load_model('./datasets/shoes_dataset/Models/model.h5')

with open('./datasets/shoes_dataset/shoes_dataset.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pkl.load(f)

preds = classifier.predict(x_test)
worst_cats = []
worst_cats_idx = []
worst_offs = []
worst_offs_idx = []
for i in range(len(y_test)):
    if y_test[i] == 0:
        worst_cats.append(preds[i])
        worst_cats_idx.append(i)
    if y_test[i] == 1:
        worst_offs.append(preds[i])
        worst_offs_idx.append(i)

im = plt.imread('./datasets/shoes_dataset/Data/Val/Shoes/100_AB.jpg')
worst_cat = im / 255.0
#worst_cat = x_test[worst_cats_idx[np.argmax(worst_cats)]]
worst_cat_pred = classifier.predict(worst_cat.reshape(1, 256, 256, 3))
worst_off = x_test[worst_offs_idx[np.argmin(worst_offs)]]
worst_off_pred = classifier.predict(worst_off.reshape(1, 256, 256, 3))

f, ax = plt.subplots(1, 2)
ax[0].imshow(worst_cat)
ax[0].set_title(worst_cat_pred)
ax[1].imshow(worst_off)
ax[1].set_title(worst_off_pred)
plt.show()

with open('./Counterfactuals/References.pkl', 'wb') as f:
    references = pkl.dump([worst_cat, worst_cat_pred, worst_off,
                           worst_off_pred], f)

print()
