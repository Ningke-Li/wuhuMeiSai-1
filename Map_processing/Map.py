import folium
import pandas as pd
import numpy as np

m = folium.Map(
    location=[-36.5, 149.0],
    zoom_start=7
)  # 维度 经度
loc = [(-37.688723, 146.44648), (-38.23858, 147.32955), (-37.688723, 147.32955), (-38.23858, 146.44648)]

hw = pd.read_csv(r"C:\Users\17382\Desktop\math\fire_nrt_M6_96619.csv")

# folium.Marker(location=[-37.790163, 148.567343], popup='<p style="color: green">this is a point</p>').add_to(m)
# folium.Marker(location=[-37.790163, 148.017149], popup='<p style="color: green">this is a point</p>').add_to(m)
# folium.Marker(location=[-37.413923, 148.567343], popup='<p style="color: green">this is a point</p>').add_to(m)
# folium.Marker(location=[-37.413923, 148.017149], popup='<p style="color: green">this is a point</p>').add_to(m)
# folium.PolyLine([
#     [-37.790163, 148.567343],
#
#     [-37.413923, 148.567343],
#
#     [-37.413923, 148.017149],
#     [-37.790163, 148.017149],
#     [-37.790163, 148.567343]
# ], color='red').add_to(m)
# count = 0
# for k in range(len(hw['longitude'])):
#     if -37.790163 <= hw['latitude'][k] <= -37.413923 and 148.017149 <= hw['longitude'][k] <= 148.567343:
#         folium.CircleMarker(location=[hw['latitude'][k], hw['longitude'][k]], radius=1).add_to(m)
#         count = count + 1
# # m.save("test2.html")
# print(count)
## count = 1365
lat = 0.376240
lon = 0.550194
number1 = 0
x = np.zeros([100, 100])
number2 = 0
for n in range(100):
    for t in range(100):
        for h in range(len(hw['longitude'])):
            # if -37.790163 <= hw['latitude'][h] <= -37.413923 and 148.017149 <= hw['longitude'][h] <= 148.567343:
            # print('1')
            # z =-37.790163 + lat * n / 100
            # print
            # 下面是求概率
            if -37.790163 + lat * n / 100 <= hw['latitude'][h] <= -37.790163 + lat * (
                    n + 1) / 100 and 148.017149 + lon * t / 100 <= hw['longitude'][
                h] <= 148.017149 + lon * (t + 1) / 100:
                # print('2')
                number1 = number1 + 1 * hw['confidence'][h] / 100
                number2 = number2 + 1 * hw['confidence'][h] / 100

            # 下面是求火焰亮度
            # if -37.790163 + lat * n / 100 <= hw['latitude'][h] <= -37.790163 + lat * (
            #         n + 1) / 100 and 148.017149 + lon * t / 100 <= hw['longitude'][
            #     h] <= 148.017149 + lon * (t + 1) / 100:
            #     number1 = number1 + 1
            #     number2 = number2 + hw['brightness'][h]

        # print(x[n, t])
        # print(number1)

        x[n, t] = number1
        number1 = 0

        # if number1 != 0:
        #     x[n, t] = number2 / number1
        # number1 = 0
        # number2 = 0
print(x)
print(number2)
p = x / number2
pd_data = pd.DataFrame(p)
pd_data.to_csv('概率.csv')
