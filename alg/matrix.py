# -*- coding:utf-8 -*-
# author: dzhhey

import pandas as pd
import numpy as np
import math


class MatrixArea:
    class SSA:
        chargeLoc = [[24, 74], [24, 24], [74, 24], [74, 74]]
        CHARGE_THRESHOLD = 0.9

        def __init__(self, name: int, matrix, loc=None):
            if loc is None:
                loc = [49, 49]
            self.loc = loc
            self.battery = 432
            self.charge_desire = 0
            self.eat_desire = 0
            self.name = name
            self.matrix = matrix
            self.ifInUrban = self.matrix.ur_df.iloc[loc[0], loc[1]]
            self.charge_left = 0

        def calc_charge_desire(self) -> float:
            def calc_d():
                d = 0
                for cLoc in MatrixArea.SSA.chargeLoc:
                    d = abs(cLoc[1] - self.loc[1]) if d < abs(cLoc[1] - self.loc[1]) else d
                    d = abs(cLoc[2] - self.loc[2]) if d < abs(cLoc[2] - self.loc[2]) else d
                return d

            dis = calc_d()
            return dis / self.battery

        def isCrowded(self) -> bool:
            if self.ifInUrban == 1:
                for drone_loc in self.matrix.SSA_loc_list:
                    d = math.sqrt((drone_loc[0] - self.loc[0]) ** 2 + (drone_loc[1] - self.loc[1]) ** 2)
                    if d == 0:
                        pass
                    else:
                        if d < 4:
                            return True
                return False
            else:
                for drone_loc in self.matrix.SSA_loc_list:
                    d = math.sqrt((drone_loc[0] - self.loc[0]) ** 2 + (drone_loc[1] - self.loc[1]) ** 2)
                    if d == 0:
                        pass
                    else:
                        if d < 10:
                            return True
                return False

        def run_away(self, scale):
            pass

        def calc_desire_and_move(self):
            if self.isCrowded():
                self.run_away(4 if self.ifInUrban == 1 else 10)
            else:
                candidates = [[self.loc[0], self.loc[1] + 1], [self.loc[0], self.loc[1] - 1],
                              [self.loc[0] - 1, self.loc[1] + 1], [self.loc[0] - 1, self.loc[1] - 1],
                              [self.loc[0] - 1, self.loc[1]],
                              [self.loc[0] + 1, self.loc[1]], [self.loc[0] + 1, self.loc[1] - 1],
                              [self.loc[0] + 1, self.loc[1] + 1]]
                max_ = 0
                win = None
                for candidate in candidates:
                    total = 0
                    for row in range(candidate[0] - 5, candidate[0] + 5):
                        for col in range(candidate[1] - 5, candidate[1] + 5):
                            total += self.matrix.ff_df.iloc(row, col) * 1000 + self.matrix.fs_df.iloc(row, col) / 100
                    win = candidate if total > max_ else win
                    max_ = total
                self.loc = win

        def charge(self):
            self.charge_left = 210

        def move(self):
            if self.charge_left == 0:
                self.charge() if self.calc_charge_desire() > MatrixArea.SSA.CHARGE_THRESHOLD \
                    else self.calc_desire_and_move()
            else:
                self.charge_left -= 1

    def __init__(self, ff_file_name, fs_file_name, ur_file_name, SSA_num, SSA_loc_list):
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
        # 初始化repeater信息（充电的地方）
        self.repeater_info = pd.DataFrame(a)
        self.repeater_info.iloc[24, 74] = 1
        self.repeater_info.iloc[24, 24] = 1
        self.repeater_info.iloc[74, 74] = 1
        self.repeater_info.iloc[74, 24] = 1

        # 初始化每个SSA对象
        self.SSA_drones = []
        for plane in range(1, SSA_num + 1):
            self.SSA_drones.append(MatrixArea.SSA(plane, matrix=self, loc=SSA_loc_list[plane-1]))
        self.SSA_loc_list = SSA_loc_list

        # 性能指标
        # 每项指标： [距离上一次观测到的时间, 观测率, 最长观测间隔]
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
    loc_list = [[24, 25], [23, 25], [24, 26], [24, 27], [24, 28],
                [29, 25], [28, 25], [27, 25], [26, 25], [25, 25]]

    m = MatrixArea('ff.csv', 'fs.csv', 'ur.csv', 10, loc_list)
    # m.show()
