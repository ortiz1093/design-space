import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(0,100,1000)
y = np.linspace(0,100,1000)

ax.fill_between(x, 40, 80, alpha=0.5, color='b')
ax.fill_betweenx(y, 40, 50, alpha=0.5, color='r')

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

plt.show()