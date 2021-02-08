# -*- coding:utf-8 -*-
# author: dzhhey

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(dpi=2000)
ax = fig.add_subplot(111)
df = pd.read_csv('../alg/result/freq/num10times400.csv')
print(df.values)

sns.heatmap(df.values, annot=False, ax=ax, linewidths=0.05)
plt.savefig('test1.jpg')
