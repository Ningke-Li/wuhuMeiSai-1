import numpy as np
import pandas as pd
import csv

from matplotlib import pyplot

hw = pd.read_csv(r"C:\Users\17382\Desktop\wuhuMeiSai\visualize\num10times400.csv")
print(hw)
x = list(range(400))
y = list(range(400))
clr = ['red','cyan','yellow','teal','coral',
       'blue','peru','slategray','pink','plum']
for i in range(10):
    for k in range(0, 400):
        x[k] = int(hw[str(k)][i][1:hw[str(k)][i].find(',')])
        y[k] = int(hw[str(k)][i][hw[str(k)][i].find(',') + 2:-1])
        # print(x)
        # print(y)
    tick_spacing = 10

    pyplot.rcParams['figure.dpi'] = 500
    pyplot.xlim(0, 100)
    pyplot.ylim(0, 100)
    pyplot.xticks(rotation=90, fontsize=5)
    pyplot.yticks(fontsize=5)
    pyplot.plot(x, y, linewidth=1, color=clr[i])
    # print(hw[str(k)][i][1:hw[str(k)][i].find(',')],hw[str(k)][i][hw[str(k)][i].find(','):-1])
    # pyplot.plot(hw[str(k)])
pyplot.show()
