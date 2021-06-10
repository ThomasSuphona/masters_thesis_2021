import matplotlib.pyplot as plt
import numpy as np

# Make some fake data and a sample plot
x = np.arange(1,100)
y = x
ax = plt.axes()
ax.plot(x,y)

# Adjust the fontsizes on tick labels for this plot


# Here is the label and arrow code of interest
ax.annotate('(a)', 
xy=(50, 50), 
xytext=(50, 51), 
xycoords='data', 
fontsize=10, 
ha='center', 
va='bottom',
arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0)
)

plt.show()