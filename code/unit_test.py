import numpy as np
from matplotlib import pyplot as plt, colors, cm

# quick and dirty test data
ext = np.linspace(0., 1., 21)
coords, _ = np.meshgrid(ext, ext)

x = coords.flatten()
y = coords.T.flatten()

vals = 1. - np.sin(coords * np.pi / 2).flatten()

# color dict
cdict = {'red': ((0., 1., 1.),
                 (1., 0., 0.)),
        'red': ((0., 1., 1.),
                 (1., 0., 0.)),
        'red': ((0., 1., 1.),
                 (1., 0., 0.)),
        'alpha': ((0., 0., 0.),
                   (1., 1., 1.))}
# colormap from dict
testcmap = colors.LinearSegmentedColormap('test', cdict)

# plotting
fig, ax = plt.subplots(1)
ax.set_facecolor('black')

ax.tripcolor(x, y, vals, cmap='test')

fig2, ax2 = plt.subplots(1)
ax2.set_facecolor('black')

ax2.scatter(x, y, c=vals, cmap='test')

plt.show()