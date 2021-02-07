# -*- coding:utf-8 -*-
# author: dzhhey

import pandas as pd
import numpy as np
import math


class MatrixArea:
    class SSA:
        chargeLoc = [[24, 74], [24, 24], [74, 24], [74, 74]]
        CHARGE_THRESHOLD = 0.9
        BASE = 8
        VIEWS = 2 * 10

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
                    d = abs(cLoc[0] - self.loc[0]) if d < abs(cLoc[0] - self.loc[0]) else d
                    d = abs(cLoc[1] - self.loc[1]) if d < abs(cLoc[1] - self.loc[1]) else d
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
            candidates = [[self.loc[0], self.loc[1] + 1], [self.loc[0], self.loc[1] - 1],
                          [self.loc[0] - 1, self.loc[1] + 1], [self.loc[0] - 1, self.loc[1] - 1],
                          [self.loc[0] - 1, self.loc[1]],
                          [self.loc[0] + 1, self.loc[1]], [self.loc[0] + 1, self.loc[1] - 1],
                          [self.loc[0] + 1, self.loc[1] + 1]]

            def calc_importance(row_, col_):
                importance = (self.matrix.ff_df.iloc[row_, col_] * 1000 + self.matrix.fs_df.iloc[
                    row_, col_] / 100) + (self.matrix.view_time_df.iloc[row_, col_]) * MatrixArea.SSA.BASE
                return importance

            # 方位1
            candidate = [self.loc[0] - 1, self.loc[1] - 1]
            importance1 = 0
            times1 = 1
            mean1 = 0
            row_min = candidate[0] - MatrixArea.SSA.VIEWS
            col_min = candidate[1] - MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, candidate[0]):
                for col in range(0 if col_min < 0 else col_min, candidate[1]):
                    times1 += 1
                    importance1 += calc_importance(row, col)
            mean1 = importance1 / times1

            # 方位2
            candidate = [self.loc[0], self.loc[1] - 1]
            importance2 = 0
            times2 = 1
            mean2 = 0
            row_min = int(candidate[0] - MatrixArea.SSA.VIEWS / 2)
            row_max = int(candidate[0] + MatrixArea.SSA.VIEWS / 2)
            col_min = candidate[1] - MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, 99 if row_max > 99 else row_max):
                for col in range(0 if col_min < 0 else col_min, candidate[1]):
                    times2 += 1
                    importance2 += calc_importance(row, col)
            mean2 = importance2 / times2

            # 方位3
            candidate = [self.loc[0] + 1, self.loc[1] - 1]
            importance3 = 0
            times3 = 1
            mean3 = 0
            row_max = candidate[0] + MatrixArea.SSA.VIEWS
            col_min = candidate[1] - MatrixArea.SSA.VIEWS
            for row in range(candidate[0], row_max if row_max < 99 else 99):
                for col in range(0 if col_min < 0 else col_min, candidate[1]):
                    times3 += 1
                    importance3 += calc_importance(row, col)
            mean3 = importance3 / times3

            # 方位4
            candidate = [self.loc[0] - 1, self.loc[1]]
            importance4 = 0
            times4 = 1
            mean4 = 0
            row_min = candidate[0] - MatrixArea.SSA.VIEWS
            col_min = int(candidate[1] - MatrixArea.SSA.VIEWS / 2)
            col_max = int(candidate[1] + MatrixArea.SSA.VIEWS / 2)
            for row in range(0 if row_min < 0 else row_min, candidate[0]):
                for col in range(0 if col_min < 0 else col_min, col_max if col_max < 99 else 99):
                    times4 += 1
                    importance4 += calc_importance(row, col)
            mean4 = importance4 / times4

            # 方位5
            candidate = [self.loc[0] + 1, self.loc[1]]
            importance5 = 0
            times5 = 1
            mean5 = 0
            row_max = candidate[0] + MatrixArea.SSA.VIEWS
            col_min = int(candidate[1] - MatrixArea.SSA.VIEWS / 2)
            col_max = int(candidate[1] + MatrixArea.SSA.VIEWS / 2)
            for row in range(candidate[0], row_max if row_max < 99 else 99):
                for col in range(0 if col_min < 0 else col_min, col_max if col_max < 99 else 99):
                    times5 += 1
                    importance5 += calc_importance(row, col)
            mean5 = importance5 / times5

            # 方位6
            candidate = [self.loc[0] - 1, self.loc[1] + 1]
            importance6 = 0
            times6 = 1
            mean6 = 0
            row_min = candidate[0] - MatrixArea.SSA.VIEWS
            col_max = candidate[1] + MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, candidate[0]):
                for col in range(candidate[1], col_max if col_max < 99 else 99):
                    times6 += 1
                    importance6 += calc_importance(row, col)
            mean6 = importance6 / times6

            # 方位7
            candidate = [self.loc[0], self.loc[1] - 1]
            importance7 = 0
            times7 = 1
            mean7 = 0
            row_min = int(candidate[0] - MatrixArea.SSA.VIEWS / 2)
            row_max = int(candidate[0] + MatrixArea.SSA.VIEWS / 2)
            col_max = candidate[1] + MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, 99 if row_max > 99 else row_max):
                for col in range(candidate[1], col_max if col_max < 99 else 99):
                    times7 += 1
                    importance7 += calc_importance(row, col)
            mean7 = importance7 / times7

            # 方位8
            candidate = [self.loc[0], self.loc[1] - 1]
            importance8 = 0
            times8 = 1
            mean8 = 0
            row_max = candidate[0] + MatrixArea.SSA.VIEWS
            col_max = candidate[1] + MatrixArea.SSA.VIEWS
            for row in range(candidate[0], 99 if row_max > 99 else row_max):
                for col in range(candidate[1], col_max if col_max < 99 else 99):
                    times8 += 1
                    importance8 += calc_importance(row, col)
            mean8 = importance8 / times8

            max_ = max(mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8)
            if mean1 == max_:
                self.loc = [self.loc[0] - 1, self.loc[1] - 1]
            if mean2 == max_:
                self.loc = [self.loc[0], self.loc[1] - 1]
            if mean3 == max_:
                self.loc = [self.loc[0] + 1, self.loc[1] - 1]
            if mean4 == max_:
                self.loc = [self.loc[0] - 1, self.loc[1]]
            if mean5 == max_:
                self.loc = [self.loc[0] + 1, self.loc[1]]
            if mean6 == max_:
                self.loc = [self.loc[0] - 1, self.loc[1] + 1]
            if mean7 == max_:
                self.loc = [self.loc[0], self.loc[1] - 1]
            if mean8 == max_:
                self.loc = [self.loc[0], self.loc[1] - 1]

        def charge(self):
            self.charge_left = 210

        def move(self):
            if self.charge_left == 0:
                self.charge() if self.calc_charge_desire() > MatrixArea.SSA.CHARGE_THRESHOLD \
                    else self.calc_desire_and_move()
            else:
                self.charge_left -= 1

    def __init__(self, ff_file_name, fs_file_name, ur_file_name, SSA_num, SSA_loc_list):
        self.ff_df = pd.read_csv(ff_file_name)  # 火灾频率
        self.ur_df = pd.read_csv(ur_file_name)  # 地形
        self.fs_df = pd.read_csv(fs_file_name)  # 火灾大小
        self.rows, self.cols = self.ff_df.shape[0], self.ff_df.shape[1]
        self.SSA_num = SSA_num

        # init time matrix
        print('col:', self.cols, ' row:', self.rows)
        a = np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows)
        self.time_df = pd.DataFrame(a)

        # init SSA and repeater
        # 所有SSA位置信息
        self.air_info = pd.DataFrame(a)
        for i in range(SSA_num):
            self.air_info.iloc[SSA_loc_list[i][0], SSA_loc_list[i][1]] = i + 1
        # 初始化repeater信息（充电的地方）
        self.repeater_info = pd.DataFrame(a)
        self.repeater_info.iloc[24, 74] = 1
        self.repeater_info.iloc[24, 24] = 1
        self.repeater_info.iloc[74, 74] = 1
        self.repeater_info.iloc[74, 24] = 1

        # 初始化每个SSA对象
        self.SSA_drones = []
        for plane in range(1, SSA_num + 1):
            self.SSA_drones.append(MatrixArea.SSA(plane, matrix=self, loc=SSA_loc_list[plane - 1]))
        self.SSA_loc_list = SSA_loc_list

        # 性能指标
        self.sf_df = pd.DataFrame(a)  # 刷新次数
        self.view_time_df = pd.DataFrame(a)  # 最近一次刷新距离现在的时间
        self.max_view_time_df = pd.DataFrame(a)  # 最长观测间隔

    def get_new_SSA_loc(self):
        new_drones_loc = []
        for drone in self.SSA_drones:
            new_drones_loc.append(drone.loc)
        self.SSA_loc_list = new_drones_loc
        print(self.SSA_loc_list)

        a = np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows)
        self.air_info = pd.DataFrame(a)
        for i in range(self.SSA_num):
            self.air_info.iloc[self.SSA_loc_list[i][0], self.SSA_loc_list[i][1]] = i + 1

    def inViews(self, point) -> bool:
        for drone in self.SSA_loc_list:
            if self.ur_df.iloc[drone[0], drone[1]] == 1:
                if math.sqrt(abs(drone[0] - point[0]) ** 2 + abs(drone[1] - point[1]) ** 2) < 4:
                    return True
            else:
                if math.sqrt(abs(drone[0] - point[0]) ** 2 + abs(drone[1] - point[1]) ** 2) < 10:
                    return True
        return False

    def refresh(self):
        for row in range(100):
            for col in range(100):
                if self.inViews([row, col]):
                    self.sf_df.iloc[row, col] += 1
                    if self.view_time_df.iloc[row, col] > self.max_view_time_df.iloc[row, col]:
                        self.max_view_time_df.iloc[row, col] = self.view_time_df.iloc[row, col]
                    self.view_time_df.iloc[row, col] = 0
                else:
                    self.view_time_df.iloc[row, col] += 1

    def next_step(self, time_):
        for plane in self.SSA_drones:
            plane.move()
        self.get_new_SSA_loc()
        self.refresh()
        self.air_info.to_csv('./result/routine/' + str(time_) + '.csv')
        print(time_)

    def start(self, times=400):
        for i in range(times):
            self.next_step(i)
        self.max_view_time_df.to_csv('./result/max_view_time/max4.csv')

    def show(self):
        print(self.ff_df)


if __name__ == '__main__':
    loc_list = [[45, 45], [45, 44], [45, 46], [45, 47], [45, 43],
                [44, 45], [44, 44], [44, 46], [44, 47], [44, 43]]

    m = MatrixArea('ff.csv', 'fs.csv', 'ur.csv', 10, loc_list)
    m.start(20)
