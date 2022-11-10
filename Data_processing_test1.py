import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import glob, os
#---------------------------------------------------------------------------------
data_processing_1 = 0
data_processing_2 = 0
draw = 1
#----------------------------------------------------------------------------------
if (data_processing_1 == 1):
    path = './'
    file = glob.glob(os.path.join(path, "AIS_2019_01_01.csv"))
    print(file)
    Data = []
    for f in file:
        data = pd.read_csv(f)
        data = data[data["MMSI"] == 316001251]
        data = data.iloc[:, :6]
        Data.append(data)
    data_oneship = pd.concat(Data)
    #------------------------------------------------------------------------------------
    data_oneship.to_csv("data_oneship_2019_01.csv")
    print(data_oneship)
elif(data_processing_2 == 1):
    data_oneship = pd.read_csv("data_oneship_2019_01.csv")
    # data_oneship['Date'] = data_oneship['BaseDateTime'].map(lambda x:x. split('T')[0])
    # data_oneship['Date'] = pd.to_datetime(data_oneship['Date'])
    # data_oneship['Time'] = data_oneship['BaseDateTime'].map(lambda x:x. split('T')[1])
    # data_oneship['Time'] = pd.to_datetime(data_oneship['Time'])
    # data_oneship = data_oneship.drop(['BaseDateTime'], axis=1)
    data_oneship['BaseDateTime'] = data_oneship['BaseDateTime'].replace("T", " ", regex=True)
    data_oneship['BaseDateTime'] = pd.to_datetime(data_oneship['BaseDateTime'])
    data_oneship = data_oneship.sort_values(by=['BaseDateTime'], ascending=True)
    data_oneship.to_csv("data_oneship_2019_01_new.csv")
elif(draw == 1):
    data_oneship = pd.read_csv("data_oneship_2019_01_new.csv")
    #-----------------------------------------------------------------------------------
    latitude = data_oneship.iloc[1].at['LAT']
    longitude = data_oneship.iloc[1].at['LON']
    location = data_oneship.values[:, 4:6].tolist()
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
    ocean_map = folium.Map(location=[latitude, longitude], zoom_start=11)
    ocean_map.add_child(incidents)

    #ocean_map.save('ocean_map_2019_01.html')
    #--------------------------------------------------------------------
    #ocean_map_route = folium.Map(location=[latitude, longitude], zoom_start=6)
    #folium.PolyLine(
    #        location,
    #        weight=1.5,
    #        color='red',
    #        opacity=0.8
    #        ).add_to(ocean_map)
    #ocean_map.save('ocean_map_route_2019_01.html')

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
draw_polylines(location, data_oneship['SOG'], ocean_map)
ocean_map.save('ocean_map_2019_01.html')
