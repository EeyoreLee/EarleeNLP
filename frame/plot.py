# -*- encoding: utf-8 -*-
'''
@create_time: 2021/02/19 14:32:03
@author: lichunyu
'''

import matplotlib.pyplot as plt
# % matplotlib inline  # notebook 内画图
import numpy as np

x = np.arange(1,5)
y = x ** 2

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14,7))
ax[0].plot(x, y)
ax[1].plot(y, x)

ax.set_title('Title',fontsize=18)
ax.set_xlabel('xlabel', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
ax.set_ylabel('ylabel', fontsize='x-large',fontstyle='oblique')
ax.legend()

pass