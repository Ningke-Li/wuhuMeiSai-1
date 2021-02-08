import csv
import pandas as pd
import numpy as np


# def step_ratio(x0):
#     n = len(x0)
#     ratio = [x0[i] / x0[i + 1] for i in range(len(x0) - 1)]
#     print(f"级比：{ratio}")
#     min_la, max_la = min(ratio), max(ratio)
#     thred_la = [np.exp(-2 / (n + 2)), np.exp(2 / (n + 2))]
#     if min_la < thred_la[0] or max_la > thred_la[-1]:
#         print("级比超过灰色模型的范围")
#     else:
#         print("级比满足要求，可用GM(1,1)模型")
#     return ratio, thred_la
#
#
# def predict(x0):
#     n = len(x0)
#     x1 = np.cumsum(x0)
#     z = np.zeros(n - 1)
#     for i in range(n - 1):
#         z[i] = 0.5 * (x1[i] + x1[i + 1])
#     B = [-z, [1] * (n - 1)]
#     Y = x0[1:]
#     u = np.dot(np.linalg.inv(np.dot(B, np.transpose(B))), np.dot(B, Y))
#     x1_solve = np.zeros(n)
#     x0_solve = np.zeros(n)
#     x1_solve[0] = x0_solve[0] = x0[0]
#     for i in range(1, n):
#         x1_solve[i] = (x0[0] - u[1] / u[0]) * np.exp(-u[0] * i) + u[1] / u[0]
#     for i in range(1, n):
#         x0_solve[i] = x1_solve[i] - x1_solve[i - 1]
#     return x0_solve, x1_solve, u


def accuracy(x0, x0_solve, ratio, u):
    epsilon = x0 - x0_solve
    delta = abs(epsilon / x0)
    print(f"相对误差：{delta}")
    # Q = np.mean(delta)
    # C = np.std(epsilon)/np.std(x0)
    S1 = np.std(x0)
    S1_new = S1 * 0.6745
    temp_P = epsilon[abs(epsilon - np.mean(epsilon)) < S1_new]
    P = len(temp_P) / len(x0)
    print(f"预测准确率：{P * 100}%")
    ratio_solve = [x0_solve[i] / x0_solve[i + 1] for i in range(len(x0_solve) - 1)]
    rho = [1 - (1 - 0.5 * u[0] / u[1]) / (1 + 0.5 * u[0] / u[1]) * (ratio[i] / ratio_solve[i]) for i in
           range(len(ratio))]
    print(f"级比偏差：{rho}")
    return epsilon, delta, rho, P


if __name__ == '__main__':
    file1 = pd.read_csv(r"C:\Users\17382\Desktop\美赛\数据\2021美赛B题澳大利亚山火数据集\fire_nrt_M6_96619.csv")
    # file1['latitude']
    data1 = '2019-10'
    k = 0
    count = [0, 0, 0, 0, 0]
    for i in range(len(file1['acq_date'])):
        # if -37.93 <= file1['latitude'][i] <= -35.2 and 146.5 <= file1['longitude'][i] <= 150.0 and file1["frp"][i] >= 30 :
        if file1["frp"][i] >= 20:
            if data1 in file1['acq_date'][i]:
                count[k] = count[k] + 1
            else:
                data1 = file1['acq_date'][i][0:7]
                k = k + 1
                count[k] = count[k] + 1
                print(count)
                print(data1)
    # print(file1['acq_date'][25][0:7])
    print(count)
    data = pd.DataFrame(data={"year": [1, 2, 3, 4], "eqL": [20541, 33828, 58732, 18720]})
    x0 = np.array(data.iloc[:, 1])
    ratio, thred_la = step_ratio(x0)
    x0_solve, x1_solve, u = predict(x0)
    epsilon, delta, rho, P = accuracy(x0, x0_solve, ratio, u)
