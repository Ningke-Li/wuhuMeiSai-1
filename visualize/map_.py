# -*- coding:utf-8 -*-
# author: dzhhey

import folium

m = folium.Map(
    location=[-36.7, 146.0],
    zoom_start=7
)  # 维度 经度
# tiles="Stamen Terrain"
m.save("test.html")