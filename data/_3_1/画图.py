import matplotlib.pyplot as plt
import numpy as np

import pandas as pd



import matplotlib.pylab as plt





data = np.load('./x_train.npy')


d = data[0]

print(d.shape)


x = d

plt.figure(figsize=(15, 10), dpi=600)
plt.xlabel("time(d)", fontsize=25)
plt.ylabel("value", fontsize=25)
plt.xticks(fontsize=20)
plt.xticks(list(range(0, 90, 10)))
plt.xlim([0, len(x)])
plt.yticks(fontsize=20)
plt.plot(x[:, 0], label='F1')
plt.plot(x[:, 1], label='F2')
plt.plot(x[:, 2], label='F3')
plt.plot(x[:, 3], label='F4')
plt.plot(x[:, 4], label='F5')
plt.plot(x[:, 5], label='F6')
plt.plot(x[:, 6], label='F7')
plt.plot(x[:, 7], label='F8')
params = {'legend.fontsize': 16}

plt.rcParams.update(params)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("./a.jpg")
# plt.show()
