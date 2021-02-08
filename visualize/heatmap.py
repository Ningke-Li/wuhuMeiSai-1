# -*- coding:utf-8 -*-
# author: dzhhey

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    df = pd.read_csv('../alg/result/freq/num26time300.csv')
    # print(df.values)
    sns.heatmap(df.values, annot=False, ax=ax, linewidths=0)
    plt.savefig('heat_num26time300.jpg')
