# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from training import dataset
import re
import sys
import projector
import training.misc as misc
import matplotlib.pyplot as plt

import pretrained_networks


sc = dnnlib.SubmitConfig()
sc.num_gpus = 1

# images = generate_images('./models/stylegan2-church-config-a.pkl',
#                          seeds=[6600, 6601],
#                          truncation_psi=0.5)

network_pkl = './models/stylegan2-church-config-a.pkl'
seed = 600
truncation_psi = 0.5
print('Loading networks from "%s"...' % network_pkl)
_G, D, Gs = pretrained_networks.load_networks(network_pkl)
proj = projector.Projector()
proj.set_network(Gs)

noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                  nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi

print('Generating image for seed %d ...' % (seed))
rnd = np.random.RandomState(seed)


initial_image = np.load('./datasets/church_val_images/church_1.npy')
initial_image_std = misc.adjust_dynamic_range(initial_image, [0, 255], [-1, 1])
proj.start([np.rollaxis(initial_image_std, 2, 0)])
latents = proj.get_dlatents()[0, 0, :].reshape(1, -1)

z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in
                noise_vars})  # [height, width]
for i in range(14):
    regenerated_image = Gs.run(latents.reshape(1, -1), None,
                               **Gs_kwargs)
    regenerated_image = np.rollaxis(regenerated_image[0, :, :, :], 2, 0)

f, ax = plt.subplots(5, 2)
ax = ax.flatten()
ax_cont = 0
ax[ax_cont].imshow(z.reshape(32, 16), cmap='binary')
ax_cont += 1
ax[ax_cont].imshow(initial_image[0, :, :, :])
ax_cont += 1
plt.pause(0.1)
for i in range(5):
    for ii in range(10):
        rn_i = np.random.randint(0, 512, 1)
        z[0, rn_i] = np.random.randn()
     # [minibatch, height, width, channel]
    new_image = Gs.run(z, None, **Gs_kwargs)
    images = np.concatenate([initial_image, new_image], axis=0)
    d_result = D.run(np.rollaxis(images, 3, 1), None)
    ax[ax_cont].imshow(z.reshape(32, 16), cmap='binary')
    ax_cont += 1
    ax[ax_cont].imshow(new_image[0, :, :, :])
    ax_cont += 1

plt.show()



print()
