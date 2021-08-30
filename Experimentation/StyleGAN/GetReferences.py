import pickle as pkl
import numpy as np
from tensorflow.keras.models import load_model

classifier = load_model('./models/classifier_model.h5')
with open('datasets/classifier_data/classifier_dataset.pkl', 'rb') as f:
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

worst_cat = x_test[worst_cats_idx[np.argmin(worst_cats)]]
worst_cat_pred = classifier.predict(worst_cat.reshape(1, 256, 256, 3))
worst_off = x_test[worst_offs_idx[4]]
worst_off_pred = classifier.predict(worst_off.reshape(1, 256, 256, 3))

with open('./Counterfactuals/References.pkl', 'wb') as f:
    references = pkl.dump([worst_cat, worst_cat_pred, worst_off,
                           worst_off_pred], f)

print()
