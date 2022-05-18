# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:38:06 2022

@author: dorey
"""

                      #### Libraries Import ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import butter, filtfilt, freqz, buttord
import os
import plotly.graph_objects as go #Library needed to plot range of axes in 3D and the slider control
import plotly.express as px #Library needed to create an animation


plt.close('all')                          # Nettoyage de la console



                         #### Reading the file ####

## Open the working file and the sheet wanted
data = pd.read_excel(r'C:\Users\dorey\Desktop\Official_test_pilot\Dat_Test_Pilote\Pilot_005_ChDo-002.xlsx', sheet_name = "Segment Position")
print (data)


                 #### Create a dataframe with the data #####

## Collect the columns names
columns = data.columns
print (columns)
print (np.size(columns))

## 
columns[1]
print(columns[1])
data.loc[0][1]
print(data.loc[0][1])
#tab = np.array
tab_2 = data.loc[0]
print(tab_2)

##
tab = np.array(tab_2)
print(tab)

## Creation of the dataframe
#REVOIR !! Ici j'ai tout taper à la main --> voir pour automatiser !
ar = np.array ([[0,0.42795376,-0.10920878,0.91235438,0.42295023,-0.12081681,0.99315467,0.42838065,-0.1167884,1.07346658,0.43023661,-0.11635898,1.14717223,0.43445544,-0.11273006,1.22060941,0.45066361,-0.09727579,1.3158499,0.49288529,-0.06927347,1.40349899,0.45743782,-0.11952142,1.2745503,0.53840058,-0.20307546,1.26270519,0.60845942,-0.24125403,0.99154271,0.7401674,-0.1687422,0.82489251,0.42923826,-0.08928744,1.27444439,0.34045032,-0.01463122,1.2595969,0.29247716,-0.02796468,0.98136097,0.42170261,0.04557128,0.8132255,0.48766358,-0.14723574,0.9128109,0.45645499,-0.14136502,0.46537181,0.41372529,-0.18499958,0.07465272,0.48619211,-0.06988638,0.01570838,0.36792489,-0.0718793,0.91644754,0.36465922,0.01871207,0.47741969,0.26146097,-0.08483145,0.11025922,0.31251422,0.03627791,0.0416807]])
df = pd.DataFrame(ar,index = ['x','y','z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','R Shoulder','L Shoulder','R U Arm','R Forearm','R Hand','L Shoulder','L U Arm','L Forearm','L Hand','R U Leg','R L Leg','R Foot','R Toe','L U Leg','L L Leg','L Foot','L Toe'])
print(df)

print(df.columns) 


#### Visualisation 3D of the data ####
plt.figure(1)  
fig.plt.figure(1)(figsize = (40,40))
ax = plt.axes (projection = '3d')
#Add title and label of the axes




##Just one example of joints link (make a loop to make it easier)
#Create the link between Pelvis and L5  'Left Forearm','Left Hand' :
ax.plot(df.loc[:,['L5','Pelvis']].loc['x',:],df.loc[:,['L5','Pelvis']].loc['y',:],df.loc[:,['L5','Pelvis']].loc['z',:])






              #### Create a figure with 3D axes ####

# https://plotly.com/python/3d-charts/

fig = go.Figure(data=[go.Mesh3d(x=(70*np.random.randn(N)),
                   y=(55*np.random.randn(N)),
                   z=(40*np.random.randn(N)),
                   opacity=0.5,
                   color='rgba(244,22,100,0.6)'
                  )])

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-100,100],),
                     yaxis = dict(nticks=4, range=[-50,100],),
                     zaxis = dict(nticks=4, range=[-100,100],),),
    width=700,
    margin=dict(r=20, l=10, b=10, t=10))



## To fixed the ratio axes
fig.update_layout(scene_aspectmode='cube')
# fix the ratio in the top left subplot to be a cube
#OR
# draw axes in proportion to the proportion of their ranges
fig.update_layout(scene3_aspectmode='data')

# To set up the axis names
fig.update_layout(scene = dict(
                    xaxis_title='X AXIS TITLE',
                    yaxis_title='Y AXIS TITLE',
                    zaxis_title='Z AXIS TITLE'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))



## To disabling tooltop spikes (keep only z axis showpikes)
fig.update_layout(scene=dict(xaxis_showspikes=False,
                             yaxis_showspikes=False))

fig.show()




              #### Create a figure with a slider control ####

fig = go.Figure()   # Create figure

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()



             #### Create an animation of the figure ####
df = px.data.gapminder()
px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
# Remplacer en conséquences - voire une fois pb df régler




            #### Save the animation ####

## Saving a GIF file
f = r"c://Users/xx/Desktop/animation.gif" 
writergif = animation.PillowWriter(fps=30) 
anim.save(f, writer=writergif)

## Saving a video file (into .avi or  .mp4 format)
f = r"c://Users/xx/Desktop/animation.mp4 (or .avi)" 
writervideo = animation.FFMpegWriter(fps=60) 
anim.save(f, writer=writervideo)






##############################################################################"
# Tentative d ouverture plus belle du fichier --> echoué pour le moment...

# #### Ouverture et import des données ####
# with open(rb'C:\Users\dorey\Desktop\Official_test_pilot\Dat_Test_Pilote\Pilot_005_ChDo-001.xlsx') as file :
#     xlsxreader = pd.read_excel(file, sheet_name = "Segment Position")
#     header    = []
#     header    = next(xlsxreader)
#     print('Name of columns :', header)
#     n_columns = len(header)
#     print('Number of columns : ', n_columns)
    
#     Frame      = []                          # Initialisation d'un vecteur vide
#     X_Position = []                          # Initialisation d'un vecteur vide
#     Y_Position = []                          # Initialisation d'un vecteur vide
#     Z_Position = []                          # Initialisation d'un vecteur vide
    
#     for row in xlsxreader: # Remplissage des vecteurs par les données du fichiers
#         Frame.append( float(row[0].replace(',', '.')))
#         X_Position.append(float(row[1].replace(',', '.')))
#         Y_Position.append(float(row[2].replace(',', '.')))
#         Z_Position.append(float(row[3].replace(',', '.')))
        
# Frame      = np.array(Frame)                       # Données Temporelles
# X_Position = np.array(X_Position)                  # Données des positions (axe X)
# X_Position = np.array(X_Position)                  # Données des positions (axe Y)
# Z_Position = np.array(Z_Position)                  # Données des positions (axe Z)

# print()