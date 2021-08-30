# Copyright 2020 Samson Woof

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %%
from utils import vgg16_mura_model, preprocess_image, show_imgwithheat
from gradcam import grad_cam, grad_cam_plus
import pickle as pkl
import matplotlib.pyplot as plt

# %% load the model
model = vgg16_mura_model('../Models/classifier_model.h5')
model.summary()
with open('../datasets/mnist_test.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pkl.load(f)

img = x_train[0, :, :, :].reshape(28, 28, 1)

heatmap_plus = grad_cam_plus(model, img, layer_name=model.layers[0].name,
                             label_name=['0', '1', '2', '3', '4', '5', '6',
                                         '7', '8', '9'])
f, ax = plt.subplots(1, 2)
ax[0].imshow(img.reshape(28, 28))
ax[1].imshow(img.reshape(28, 28))
ax[1].imshow(heatmap_plus, alpha=0.4, cmap='hot')

plt.show()