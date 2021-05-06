import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
u = -1
v = -1
f = ax.quiver([0.5], [0.5], [u], [v], angles='xy', scale_units='xy', scale=0.1)


print(np.mod(1, 2))
ax.set_xlim([-1, 2])
ax.set_ylim([0, 2])
