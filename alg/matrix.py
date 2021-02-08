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
            self.routine = []

        def calc_charge_desire(self):
            def calc_d():
                d_list = []
                d = 0
                for cLoc in MatrixArea.SSA.chargeLoc:
                    d = abs(cLoc[0] - self.loc[0]) if d < abs(cLoc[0] - self.loc[0]) else d
                    d = abs(cLoc[1] - self.loc[1]) if d < abs(cLoc[1] - self.loc[1]) else d
                    d_list.append(d)
                distance = d_list.index(d)
                return d, MatrixArea.SSA.chargeLoc[distance]

            dis, location = calc_d()
            return location, dis / self.battery

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
                    row_, col_] / 100 + MatrixArea.SSA.BASE) * (self.matrix.view_time_df.iloc[row_, col_])
                return importance

            # 反群聚
            def run_away(point):
                for drone in self.matrix.SSA_drones:
                    if drone.loc == point:
                        return True
                return False

            # 方位1
            candidate = [min(self.loc[0] - 1, 0), min(self.loc[1] - 1, 0)]
            importance1 = 0
            times1 = 1
            mean1 = 0
            row_min = candidate[0] - MatrixArea.SSA.VIEWS
            col_min = candidate[1] - MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, candidate[0]):
                for col in range(0 if col_min < 0 else col_min, candidate[1]):
                    times1 += 1
                    importance1 += calc_importance(row, col)
            mean1 = - importance1 if run_away([min(self.loc[0] - 1, 0), min(self.loc[1] - 1, 0)]) else importance1

            # 方位2
            candidate = [self.loc[0], max(0, self.loc[1] - 1)]
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
            mean2 = - importance2 if run_away([self.loc[0], max(0, self.loc[1] - 1)]) else importance2

            # 方位3
            candidate = [min(self.loc[0] + 1, 99), max(0, self.loc[1] - 1)]
            importance3 = 0
            times3 = 1
            mean3 = 0
            row_max = candidate[0] + MatrixArea.SSA.VIEWS
            col_min = candidate[1] - MatrixArea.SSA.VIEWS
            for row in range(candidate[0], row_max if row_max < 99 else 99):
                for col in range(0 if col_min < 0 else col_min, candidate[1]):
                    times3 += 1
                    importance3 += calc_importance(row, col)
            mean3 = - importance3 if run_away([min(self.loc[0] + 1, 99), max(0, self.loc[1] - 1)]) else importance3

            # 方位4
            candidate = [max(self.loc[0] - 1, 0), self.loc[1]]
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
            mean4 = - importance4 if run_away([max(self.loc[0] - 1, 0), self.loc[1]]) else importance4

            # 方位5
            candidate = [max(self.loc[0] + 1, 99), self.loc[1]]
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
            mean5 = - importance5 if run_away([max(self.loc[0] + 1, 99), self.loc[1]]) else importance5

            # 方位6
            candidate = [max(self.loc[0] - 1, 0), min(self.loc[1] + 1, 99)]
            importance6 = 0
            times6 = 1
            mean6 = 0
            row_min = candidate[0] - MatrixArea.SSA.VIEWS
            col_max = candidate[1] + MatrixArea.SSA.VIEWS
            for row in range(0 if row_min < 0 else row_min, candidate[0]):
                for col in range(candidate[1], col_max if col_max < 99 else 99):
                    times6 += 1
                    importance6 += calc_importance(row, col)
            mean6 = - importance6 if run_away([max(self.loc[0] - 1, 0), min(self.loc[1] + 1, 99)]) else importance6

            # 方位7
            candidate = [self.loc[0], min(self.loc[1] + 1, 99)]
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
            mean7 = - importance7 if run_away([self.loc[0], min(self.loc[1] + 1, 99)]) else importance7

            # 方位8
            candidate = [min(self.loc[0] + 1, 99), min(self.loc[1] + 1, 99)]
            importance8 = 0
            times8 = 1
            mean8 = 0
            row_max = candidate[0] + MatrixArea.SSA.VIEWS
            col_max = candidate[1] + MatrixArea.SSA.VIEWS
            for row in range(candidate[0], 99 if row_max > 99 else row_max):
                for col in range(candidate[1], col_max if col_max < 99 else 99):
                    times8 += 1
                    importance8 += calc_importance(row, col)
            mean8 = - importance8 if run_away([min(self.loc[0] + 1, 99), min(self.loc[1] + 1, 99)]) else importance8

            max_ = max(mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8)
            if mean1 == max_:
                self.loc = [max(self.loc[0] - 1, 0), max(self.loc[1] - 1, 0)]
            if mean2 == max_:
                self.loc = [self.loc[0], max(0, self.loc[1] - 1)]
            if mean3 == max_:
                self.loc = [min(self.loc[0] + 1, 99), max(0, self.loc[1] - 1)]
            if mean4 == max_:
                self.loc = [max(self.loc[0] - 1, 0), self.loc[1]]
            if mean5 == max_:
                self.loc = [min(self.loc[0] + 1, 99), self.loc[1]]
            if mean6 == max_:
                self.loc = [max(self.loc[0] - 1, 0), min(self.loc[1] + 1, 99)]
            if mean7 == max_:
                self.loc = [self.loc[0], min(self.loc[1] + 1, 99)]
            if mean8 == max_:
                self.loc = [min(self.loc[0] + 1, 99), min(self.loc[1] + 1, 99)]
            if max_ < 0:
                self.loc = self.loc

        def charge(self, loc):
            self.charge_left = 210

        def move(self):
            charge_Candidate, cDesire = self.calc_charge_desire()
            if self.charge_left == 0:
                self.charge(charge_Candidate) if cDesire > MatrixArea.SSA.CHARGE_THRESHOLD \
                    else self.calc_desire_and_move()
            else:
                self.charge_left -= 1
                self.loc = charge_Candidate
            self.routine.append(str(self.loc))

    def __init__(self, ff_file_name, fs_file_name, ur_file_name, SSA_num, SSA_loc_list):
        self.ff_df = pd.read_csv(ff_file_name)  # 火灾频率
        self.ur_df = pd.read_csv(ur_file_name)  # 地形
        self.fs_df = pd.read_csv(fs_file_name)  # 火灾大小
        self.rows, self.cols = self.ff_df.shape[0], self.ff_df.shape[1]
        self.SSA_num = SSA_num

        # init time matrix
        print('col:', self.cols, ' row:', self.rows)
        self.time_df = pd.DataFrame(np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))

        # init SSA and repeater
        # 所有SSA位置信息
        self.air_info = pd.DataFrame(np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))
        for i in range(SSA_num):
            self.air_info.iloc[SSA_loc_list[i][0], SSA_loc_list[i][1]] = i + 1
        # 初始化repeater信息（充电的地方）
        self.repeater_info = pd.DataFrame(
            np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))
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
        self.sf_df = pd.DataFrame(
            np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))  # 刷新次数
        self.view_time_df = pd.DataFrame(
            np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))  # 最近一次刷新距离现在的时间
        self.max_view_time_df = pd.DataFrame(
            np.array([0 for i in range(self.cols * self.rows)]).reshape(self.cols, self.rows))  # 最长观测间隔

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
        # self.air_info.to_csv('./result/routine/' + str(time_) + '.csv')
        print(time_)

    def start(self, times=400):
        for i in range(times):
            self.next_step(i)

        self.max_view_time_df.to_csv('./result/max_view_time/num' + str(self.SSA_num) + 'times' + str(times) + '.csv')
        self.sf_df.to_csv('./result/freq/num' + str(self.SSA_num) + 'time' + str(times) + '.csv')
        routine_list = []
        for i in range(self.SSA_num):
            routine_list.append(self.SSA_drones[i].routine)
        r_df = pd.DataFrame(routine_list, index=['plane' + str(i + 1) for i in range(self.SSA_num)])
        r_df.to_csv('./result/routine/num' + str(self.SSA_num) + 'times' + str(times) + '.csv')
        print(self.calc_performance(times))

    def show(self):
        print(self.ff_df)

    def calc_performance(self, times):
        performance = 0

        def calc_importance(row_, col_):
            importance = (self.ff_df.iloc[row_, col_] * 1000 + self.fs_df.iloc[
                row_, col_] / 100 + 2) * (self.sf_df.iloc[row_, col_] / times)
            return importance

        for row in range(100):
            for col in range(100):
                performance += calc_importance(row, col)
        return performance


if __name__ == '__main__':
    loc_list = [[45, 45], [45, 44], [45, 46], [45, 47], [45, 43],
                [44, 45], [44, 44], [44, 46], [44, 47], [44, 43],
                [42, 45], [42, 44], [42, 46], [42, 47], [42, 43],
                [43, 45], [43, 44], [43, 46], [43, 47], [43, 43],
                [46, 43], [46, 44], [46, 45], [46, 46], [46, 42],
                [41, 43], [41, 44], [41, 45], [41, 46], [41, 42],
                [47, 43], [47, 44], [47, 45], [47, 46], [47, 42],
                [47, 43], [47, 44], [47, 45], [47, 46], [47, 42],
                [47, 43], [47, 44], [47, 45], [47, 46], [47, 42],
                [47, 43], [47, 44], [47, 45], [47, 46], [47, 42]]

    num = 50
    m = MatrixArea('ff.csv', 'fs.csv', 'ur.csv', num, loc_list[:num])
    print(str(num) + 'bu')
    m.start(300)
