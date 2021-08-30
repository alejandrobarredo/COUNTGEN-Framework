import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gradCam.gradcam import grad_cam_plus
from tensorflow.keras.models import load_model


def min_max(_x, _min=None, _max=None):
    if _min is None and _max is None:
        return (_x - np.min(_x)) / (np.max(_x) - np.min(_x))
    else:
        return (_x - _min) / (_max - _min)


def hide_border(_ax):
    right_side = _ax.spines["right"]
    top_side = _ax.spines["top"]
    bot_side = _ax.spines["bottom"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    bot_side.set_visible(False)


def plot_front(_ax, _x, _y, _z):
    _ax.set_xlabel('Code D')
    _ax.set_ylabel('Prediction')
    _ax.set_zlabel('Plausibility')
    _cs = _ax.scatter3D(_x, _y, _z, c=(_x + _y + _z))
    _ax.set_xlim(_ax.get_xlim()[::-1])
    plt.colorbar(_cs, orientation="horizontal",
                 pad=0.01, ax=_ax)
    plt.pause(0.1)
    return _cs.cmap


def plot_counterfactuals(_cax, _fax, _sax, _hax, _x, _y, _z, _i, _c):
    count_idx = np.linspace(start=0, stop=len(_i) - 1, num=8,
                            dtype=np.int)

    highlight_cmap = cm.get_cmap('Set3')
    for i, idx in enumerate(count_idx):
        heatmap = grad_cam_plus(_c, _i[idx].reshape(256, 256, 3),
                                layer_name=_c.layers[0].name,
                                label_name=['0', '1', '2', '3', '4', '5',
                                            '6', '7', '8', '9'])
        _cax[i + 2].imshow(_i[idx].reshape(256, 256, 3))
        _sax[i + 2].barh(['Code D', 'Prediction', 'Plausibility'],
                         [_x[idx], _y[idx], _z[idx]],
                         align='center')
        # _hax[i + 2].imshow(_i[idx].reshape(256, 256))
        _hax[i + 2].imshow(heatmap, alpha=1, cmap='hot',
                           vmin=np.min(heatmap),
                           vmax=np.max(heatmap))

        if (_y[idx] > 0.5) and (_z[idx] > 0.5):
            _sax[i + 2].set_title('Plausible')
        plt.pause(0.1)
        _fax.scatter3D(_x[idx], _y[idx], _z[idx],
                       c=highlight_cmap(i),
                       s=100,
                       marker='X')
        plt.pause(0.1)


def plot_references(_cax, _fax, _sax, _hax, _refs, _c):
    worst_four = _refs[0]
    worst_four_pred = _refs[1]
    worst_eight = _refs[2]
    worst_eight_pred = _refs[3]

    worst_four_heatmap = grad_cam_plus(_c,
                                       worst_four.reshape(256, 256, 3)/255.0,
                                       layer_name=_c.layers[0].name,
                                       label_name=['0', '1', '2', '3', '4',
                                                   '5', '6', '7', '8', '9'])
    worst_eight_heatmap = grad_cam_plus(_c,
                                        worst_eight.reshape(256, 256, 3)/255,
                                       layer_name=_c.layers[0].name,
                                       label_name=['0', '1', '2', '3', '4',
                                                   '5', '6', '7', '8', '9'])
    _cax[0].imshow(worst_four.reshape(256, 256, 3))
    _sax[0].barh([i for i in range(2)],
                 [worst_four_pred[0][0], 1 - worst_four_pred[0][0]],
                 align='center')
    _sax[0].set_yticks([i for i in range(2)])

    _sax[0].set_xlim(0.001, 0.999)
    _sax[0].set_title('Seed')
    #_hax[0].imshow(worst_four.reshape(256, 256))
    _hax[0].imshow(worst_four_heatmap, alpha=1, cmap='hot',
                   vmin=np.min(worst_four_heatmap),
                   vmax=np.max(worst_four_heatmap))

    _cax[1].imshow(worst_eight.reshape(256, 256, 3))
    _sax[1].barh([i for i in range(2)],
                 [worst_eight_pred[0][0], 1 - worst_eight_pred[0][0]],
                 align='center')
    _sax[1].set_yticks([i for i in range(2)])

    _sax[1].set_xlim(0.001, 0.999)
    _sax[1].set_title('Worst opposite')

    #_hax[1].imshow(worst_eight.reshape(256, 256))
    _hax[1].imshow(worst_eight_heatmap, alpha=1, cmap='hot',
                   vmin=np.min(worst_eight_heatmap),
                   vmax=np.max(worst_eight_heatmap))
    print()


with open('./Counterfactuals/Front_0.pkl', 'rb') as f:
    front = pkl.load(f)

with open('./Counterfactuals/References.pkl', 'rb') as f:
    references = pkl.load(f)

classifier = load_model('./datasets/shoes_dataset/Models/model.h5')

x_data = []
y_data = []
z_data = []
image_data = []

for sol in front:
    x_data.append(sol.objectives[0])
    y_data.append(np.squeeze(sol.objectives[1]))
    z_data.append(sol.objectives[2])
    image_data.append(sol.counterfactualImage)

x_data = min_max(x_data)
y_data = 1 - min_max(y_data)
z_data = 1 - min_max(z_data)

fig = plt.figure(constrained_layout=True)
spec2 = fig.add_gridspec(5, 30)
front_ax = fig.add_subplot(spec2[0:5, 0:12], projection='3d')
n_row = 12
n_col = 20
counter_spec = gridspec.GridSpecFromSubplotSpec(n_row, n_col,
                                                subplot_spec=spec2[0:5, 13:256])
counterfactuals_ax = []
heatmap_ax = []
stats_ax = []
for col in range(5):
    col_x = col*4
    ax = plt.Subplot(fig, counter_spec[0:2, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    counterfactuals_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[3:5, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    heatmap_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[5:6, col_x:col_x+4])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim(0, 1)
    hide_border(ax)
    fig.add_subplot(ax)
    stats_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[6:8, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    counterfactuals_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[9:11, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    heatmap_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[11:12, col_x:col_x+4])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim(0, 1)
    hide_border(ax)
    fig.add_subplot(ax)
    stats_ax.append(ax)

plt.pause(0.1)
print()

# PLOT References
plot_references(counterfactuals_ax, front_ax, stats_ax, heatmap_ax,
                references, classifier)

# PLOT front
custom_cmap = plot_front(front_ax, x_data, y_data, z_data)
plt.pause(0.1)

# PLOT Counterfactuals
plot_counterfactuals(counterfactuals_ax, front_ax, stats_ax, heatmap_ax, x_data, y_data,
                     z_data, image_data, classifier)
plt.pause(0.1)

plt.show()

print()