# -*- coding:utf-8 -*-
# author: dzhhey

import folium

m = folium.Map(
    location=[-37, 147.5],
    zoom_start=7
)  # 维度 经度
# tiles="Stamen Terrain"
m.save("test.html")
