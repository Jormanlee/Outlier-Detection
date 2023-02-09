import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import folium
import glob, os

def draw_polylines(points, speeds, map):
    colors = [speed_color(x) for x in speeds]
    n = len(colors)
    # Need to have a corresponding color for each point
    if n != len(points):
        raise ValueError
    i = 0
    j = 1
    curr = colors[0]
    while i < n and j < n:
        if colors[i] != colors[j]:
            line = folium.PolyLine(points[i:j], color=curr, weight=2.5, opacity=1)
            line.add_to(map)
            curr = colors[j]
            i = j
        j += 1
    if i < j:
        folium.PolyLine(points[i:j], color=curr, weight=2.5, opacity=1).add_to(map)


def speed_color(speed):
    if speed < 0:
        raise ValueError
    elif speed >= 0 and speed < 10:
        return 'green'
    elif speed >= 10 and speed < 18:
        return 'yellow'
    else:
        return 'red'

Total_Number_ship = 15
Number_ship = 1
Path = "../Input_2/Generation_data/Train_data_2/"
Path_map = "../Input_2/Map/Generation_train_map_2/"
filename_draw = (Path + "generation_data_%d.csv" %(Number_ship))
data_oneship = pd.read_csv(filename_draw)
mapname_draw = (Path_map +"map_generation_data_%d_ships.html" %(Total_Number_ship))
latitude = data_oneship.iloc[1].at['LAT']
longitude = data_oneship.iloc[1].at['LON']
ocean_map = folium.Map(location=[latitude, longitude], zoom_start=11)
for Number_ship in range(1, Total_Number_ship+1):
    filename_draw = (Path + "generation_data_%d.csv" %(Number_ship))
    data_oneship = pd.read_csv(filename_draw)
    # data_oneship['BaseDateTime'] = pd.to_datetime(data_oneship['BaseDateTime'])
    # data_oneship.set_index('BaseDateTime', inplace=True)
    #-----------------------------------------------------------------------------------
    location = data_oneship.values[:, 1:3].tolist()
    #------------------------------------------------------------------------------------
    # Instantiate a feature group for the incidents in the dataframe
    incidents = folium.map.FeatureGroup()

    # Loop through the each data point and add each to the incidents feature group
    # 2019 01 
    for lat, lng, in zip(data_oneship.LAT, data_oneship.LON):
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=0.05, # define how big you want the circle markers to be
                color='black',
                fill=True,
                fill_color='red',
                fill_opacity=0.4
            )
        )

    # Add incidents to map
    ocean_map.add_child(incidents)
    draw_polylines(location, data_oneship['SOG'], ocean_map)
ocean_map.save(mapname_draw)
