import csv
import math

import matplotlib
from folium import plugins
from matplotlib import pyplot
import numpy as np
import pandas as pd
import folium

m = folium.Map(
    location=[-36.5, 149.0],
    zoom_start=7
)  # 维度 经度

def degree_conversion_decimal(x):
    """
    度分转换成十进制
    :param x: float
    :return: integer float
    """
    integer = int(x)
    integer = integer + (x - integer) * 1.66666667
    return integer


def distance(origin, destination):
    """
    经纬度计算两点距离
    :param origin:
    :param destination:
    :return:
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

hw = pd.read_csv(r"C:\Users\17382\Desktop\美赛\数据\2021美赛B题澳大利亚山火数据集\fire_nrt_M6_96619.csv")
#这部分是画不包括地图信息的点图
for x in range(len(hw['longitude'])):
    if (-37.93 <= hw['latitude'][x] <= -35.2 and 146.5 <= hw['longitude'][x] <= 150.0):
        pyplot.scatter(hw['latitude'][x],hw['longitude'][x],s=hw['bright_t31'][x]/100)
        # print('1')
        print(x)
pyplot.show()
#
# hw['latitude'] = hw['latitude'].apply(degree_conversion_decimal)
# hw['longitude'] = hw['longitude'].apply(degree_conversion_decimal)
#以下是将地图加点打印出来
# k=0
#
# lction = [hw['latitude'],hw['longitude']]
# for k in range(len(hw['longitude'])):
#     if(-39.2<=hw['latitude'][k]<=-34.7 and 143.2<=hw['longitude'][k]<=150.0):
#         folium.CircleMarker(location=[hw['latitude'][k],hw['longitude'][k]],radius=1).add_to(m)
#
# # folium.Marker(location).add_to(m)
# m.save("test1.html")