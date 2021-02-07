# -*- coding:utf-8 -*-
# author: dzhhey

import pandas as pd
import numpy as np


class MatrixArea:
    class SSA:
        def __init__(self, name: int, matrix, loc=None):
            if loc is None:
                loc = [49, 49]
            self.loc = loc
            self.battery = 1.0
            self.charge_desire = 0
            self.eat_desire = 0
            self.name = name
            self.matrix = matrix

        def calc_charge_desire(self) -> float:
            pass

        def calc_eat_desire(self) -> float:
            pass

        def charge(self):
            pass

        def eat(self):
            pass

        def move(self):
            self.charge() if self.calc_eat_desire() > self.calc_charge_desire() else self.eat()

    def __init__(self, ff_file_name, fs_file_name, ur_file_name, SSA_num):
        self.ff_df = pd.read_csv(ff_file_name)
        self.ur_df = pd.read_csv(ur_file_name)
        self.fs_df = pd.read_csv(fs_file_name)
        self.rows, self.cols = self.ff_df.shape[0], self.ff_df.shape[1]

        # init time matrix
        print('col:', self.cols, ' row:', self.rows)
        a = np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows)
        self.time_df = pd.DataFrame(a)

        # init SSA and repeater
        # 所有SSA位置信息
        self.air_info = pd.DataFrame(a)
        self.air_info.iloc[49, 49] = str([i for i in range(1, SSA_num + 1)])
        # print(self.air_info.iloc[45: 55, 45: 55])

        # 初始化每个SSA对象
        self.SSA_drones = []
        for plane in range(1, SSA_num + 1):
            self.SSA_drones.append(MatrixArea.SSA(plane, matrix=self))

        # 性能指标
        # 每项指标： [距离上一次观测到的时间, 观测率, 超时次数]
        b = np.array([str([0, 0, 0]) for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows)
        self.sf_df = pd.DataFrame(b)  # 刷新率

    def get_new_SSA_loc(self) -> pd.DataFrame:
        pass

    def next_step(self):
        for plane in self.SSA_drones:
            plane.move()
        self.get_new_SSA_loc()

    def show(self):
        print(self.ff_df)


if __name__ == '__main__':
    m = MatrixArea('ff.csv', 'fs.csv', 'ur.csv', 10)
    # m.show()
