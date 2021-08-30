import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def min_max(_x, _min=None, _max=None):
    if _min is None and _max is None:
        return (_x - np.min(_x)) / (np.max(_x) - np.min(_x))
    else:
        return (_x - _min) / (_max - _min)


def plot_front(_ax, _c_ax, _x, _y, _z):
    _ax.set_xlabel('Code D')
    _ax.set_ylabel('Prediction')
    _ax.set_zlabel('Plausibility')
    _cs = _ax.scatter3D(_x, _y, _z, c=(_x + _y + _z))
    plt.colorbar(_cs,
                 orientation="horizontal", pad=0.2, ax=_c_ax)
    return _cs.cmap


def plot_data(_ax, _l_ax, _x, _y, _z, _cmap):
    _ax.set_ylim((0, np.max(_x + _y + _x) * 1.1))
    _ax.scatter(range(100), _x, s=1, c='red',
                alpha=0.3)
    l1, = _ax.plot(_x, c='red', label='Code D (X)', alpha=0.3)

    _ax.scatter(range(100), _y, s=1, c='green',

                alpha=0.3)
    l2, = _ax.plot(_y, c='green', label='Prediction (Y)', alpha=0.3)

    _ax.scatter(range(100), _z, s=1, c='blue',

                alpha=0.3)
    l3, = _ax.plot(_z, c='blue', alpha=0.3, label='Plausibility (Z)')

    _ax.scatter(range(100), _x + _y + _z,
                s=2,
                marker='x',
                c=(_x + _y + _z),
                cmap=_cmap)
    l4, = _ax.plot(_x + _y + _z,
                   c='black',
                   label='X + Y + Z',
                   alpha=0.3)
    _l_ax.legend(handles=[l1, l2, l3, l4],
                 fontsize='xx-small')


with open('./Counterfactuals/Front_2.pkl', 'rb') as f:
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

x_data = min_max(x_data).reshape(-1)
y_data = min_max(y_data).reshape(-1)
z_data = min_max(z_data).reshape(-1)

fig = plt.figure(constrained_layout=True)
spec2 = fig.add_gridspec(5, 17)
front_ax = fig.add_subplot(spec2[0:3, 0:6], projection='3d')
data_ax = fig.add_subplot(spec2[3:5, 0:16])
legend_ax = fig.add_subplot(spec2[3:5, 16])
legend_ax.axis('off')
legend_ax.set_xticks([])
legend_ax.set_yticks([])
n_row = 2
n_col = 5
counter_spec = gridspec.GridSpecFromSubplotSpec(n_row, n_col,
                                                subplot_spec=spec2[0:3, 7:17])
counterfactuals_ax = []
for col in range(n_col):
    for row in range(n_row):
        ax = plt.Subplot(fig, counter_spec[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
        counterfactuals_ax.append(ax)

# PLOT front
custom_cmap = plot_front(front_ax, data_ax, x_data, y_data, z_data)

# PLOT data
plot_data(data_ax, legend_ax, x_data, y_data, z_data, custom_cmap)

# PLOT Counterfactuals
count_idx = np.linspace(start=0, stop=len(image_data) - 1, num=10,
                        dtype=np.int)

highlight_cmap = cm.get_cmap('Set3')
for i, idx in enumerate(count_idx):
    value = x_data[idx] + y_data[idx] + z_data[idx]

    counterfactuals_ax[i].imshow(image_data[idx].reshape(256, 256, 3))

    plt.setp(counterfactuals_ax[i].spines.values(),
             color=highlight_cmap(i),
             linewidth=4)
    data_ax.axvline(x=idx, linestyle='-.', alpha=0.1)
    data_ax.scatter(idx, value,
                    s=30,
                    marker='X',
                    c=highlight_cmap(i))
    front_ax.scatter3D(x_data[idx], y_data[idx], z_data[idx],
                       c=highlight_cmap(i),
                       s=100,
                       marker='X')
    plt.pause(0.1)
plt.show()

print()