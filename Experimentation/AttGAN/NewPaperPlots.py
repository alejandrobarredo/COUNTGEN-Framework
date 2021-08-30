import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import silent_list
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
    plt.colorbar(_cs, orientation="horizontal",
                 pad=0.01, ax=_ax)
    plt.pause(0.1)
    return _cs.cmap


def plot_counterfactuals(_cax, _fax, _sax, _x, _y, _z, _i):
    # count_idx = np.linspace(start=0, stop=len(_i) - 1, num=8,
    #                         dtype=np.int)
    count_idx = np.random.choice(list(range(len(_i))), size=8)

    highlight_cmap = cm.get_cmap('Set3')
    for i, idx in enumerate(count_idx):
        _cax[i + 2].imshow(min_max(_i[idx]))
        _sax[i + 2].barh(['Code D', 'Prediction', 'Plausibility'],
                         [_x[idx], _y[idx], _z[idx]],
                         align='center')
        if (_y[idx] < 0.5) and (_z[idx] < 0.5):
            _sax[i + 2].set_title('Counterfactual')
        plt.pause(0.1)
        _fax.scatter3D(_x[idx], _y[idx], _z[idx],
                       c=highlight_cmap(i),
                       s=100,
                       marker='X')
        plt.pause(0.1)


def plot_references(_cax, _fax, _sax, _refs):
    worst_four = _refs[0]
    worst_four_pred = _refs[1]
    worst_eight = _refs[2]
    worst_eight_pred = _refs[3]
    classes = len(worst_four_pred)
    _cax[0].imshow(worst_four)
    _sax[0].barh([i for i in range(2)],
                 [worst_four_pred[0][0], 1 - worst_four_pred],
                 align='center')
    _sax[0].set_yticks([i for i in range(2)])
    #_sax[0].set_xscale("log")
    _sax[0].set_title('Wost Man')

    _cax[1].imshow(worst_eight)
    _sax[1].barh([i for i in range(2)],
                 [worst_eight_pred[0][0], 1 - worst_eight_pred],
                 align='center')
    _sax[1].set_yticks([i for i in range(2)])
    #_sax[1].set_xscale("log")
    _sax[1].set_title('Worst Female')
    plt.pause(0.1)
    print()


with open('./Counterfactuals/Front_0.pkl', 'rb') as f:
    front = pkl.load(f)


x_data = []
y_data = []
z_data = []
image_data = []

for sol in front:
    x_data.append(np.squeeze(sol.objectives[0]))
    y_data.append(np.squeeze(sol.objectives[1]))
    z_data.append(np.squeeze(sol.objectives[2]))
    image_data.append(sol.counterfactualImage)

x_data = min_max(x_data)
y_data = min_max(y_data)
z_data = min_max(z_data)

fig = plt.figure(constrained_layout=True)
spec2 = fig.add_gridspec(5, 30)
front_ax = fig.add_subplot(spec2[0:5, 0:12], projection='3d')
n_row = 12
n_col = 20
counter_spec = gridspec.GridSpecFromSubplotSpec(n_row, n_col,
                                                subplot_spec=spec2[0:5, 13:28])
counterfactuals_ax = []
stats_ax = []
for col in range(5):
    col_x = col*4
    ax = plt.Subplot(fig, counter_spec[0:5, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    counterfactuals_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[5:6, col_x:col_x+4])
    ax.set_xticks([0, 0.5, 1])
    hide_border(ax)
    fig.add_subplot(ax)
    stats_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[6:11, col_x:col_x+4])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    counterfactuals_ax.append(ax)

    ax = plt.Subplot(fig, counter_spec[11:12, col_x:col_x+4])
    ax.set_xticks([0, 0.5, 1])
    hide_border(ax)
    fig.add_subplot(ax)
    stats_ax.append(ax)

plt.pause(0.1)
print()

# PLOT References
with open('./Counterfactuals/References.pkl', 'rb') as f:
    references = pkl.load(f)
plot_references(counterfactuals_ax, front_ax, stats_ax, references)

# PLOT front
custom_cmap = plot_front(front_ax, x_data, y_data, z_data)
plt.pause(0.1)

# PLOT Counterfactuals
plot_counterfactuals(counterfactuals_ax, front_ax, stats_ax, x_data, y_data,
                     z_data, image_data)
plt.pause(0.1)

plt.show()

print()