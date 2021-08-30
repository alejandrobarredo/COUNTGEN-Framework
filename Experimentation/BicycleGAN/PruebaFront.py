import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from jMetalPy.jmetal.lab.visualization import Plot

with open('/home/xai/Documentos/BicycleGAN/datasets'
          '/shoes_dataset/Counterfactuals/Front.pkl', 'rb') as f:
    front = pkl.load(f)
x_data = []
y_data = []
z_data = []
image_data = []
for sol in front:
    x_data.append(sol.objectives[0])
    y_data.append(np.squeeze(sol.objectives[1]))
    z_data.append(sol.objectives[2])
    image_data.append(sol.counterfactualImage)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Code D')
ax.set_ylabel('Prediction')
ax.set_zlabel('Plausibility')
ax.scatter3D(x_data, y_data, z_data, cmap='Greens')

fig, ax_image = plt.subplots(10, 10)
for i in range(10):
    for ii in range(10):
        ax_image[i, ii].imshow(image_data[i*1 + i*10])
        ax_image[i, ii].axis('off')

plt.show()
print()