#!/usr/bin/env python
# coding: utf-8

# Importing Libraries :

# In[330]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Functions Defenition :

# In[331]:


def plot_all_sensors_with_names (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=df.loc[:,sensor_name][0]
        y=df.loc[:,sensor_name][1]
        z=df.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (sensor_name), size=20, zorder=1,)


# In[332]:


def plot_all_sensors_with_numbers (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=df.loc[:,sensor_name][0]
        y=df.loc[:,sensor_name][1]
        z=df.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (i+1), size=30, zorder=1,)


# In[333]:


def plot_all_segments(data,ax) :
    for i in range (0,len(segments_right)) :
        ax.plot(data.loc[:,segments_right[i]].loc['x',:],data.loc[:,segments_right[i]].loc['y',:],data.loc[:,segments_right[i]].loc['z',:], color="b")
    for i in range (0,len(segments_left)) :
        ax.plot(data.loc[:,segments_left[i]].loc['x',:],data.loc[:,segments_left[i]].loc['y',:],data.loc[:,segments_left[i]].loc['z',:], color="r")
    for i in range (0,len(segments_axial)) :
        ax.plot(data.loc[:,segments_axial[i]].loc['x',:],data.loc[:,segments_axial[i]].loc['y',:],data.loc[:,segments_axial[i]].loc['z',:], color="k")


# Reading movement stairs data :

# In[334]:


data=pd.read_excel('Pilot_006_JeEy_Stairs.xlsx', sheet_name ="Segment Position")
data


# In[335]:


columns= data.columns
columns


# Preparing 3 arrays : -- segments_right contains the segments of the right limbs -- segments_left contains the segments of the left limbs -- segments_axial contains the segments of the axial part

# In[336]:


segments_right=np.array([['Neck', 'Right Shoulder'],['Right Upper Arm', 'Right Shoulder'],['Right Forearm','Right Hand'],['Right Forearm','Right Upper Arm'],['Right Upper Leg', 'Right Lower Leg'],['Right Upper Leg', 'Pelvis'],['Right Foot', 'Right Lower Leg'],['Right Foot', 'Right Toe']])
segments_left=np.array([['Left Forearm','Left Hand'],['Left Forearm','Left Upper Arm'],['Neck', 'Left Shoulder'],['Left Upper Arm', 'Left Shoulder'],['Left Upper Leg', 'Pelvis'],['Left Foot', 'Left Lower Leg'],['Left Foot', 'Left Toe'],['Left Upper Leg', 'Left Lower Leg']])
segments_axial=np.array([['Neck', 'Head'],['T8', 'Neck'],['T12', 'T8'],['L3', 'T12'],['L3', 'L5'],['L5', 'Pelvis']])


# In[337]:


print('segments_right :')
print(segments_right)
print('\nsegments_left :')
print(segments_left)
print('\nsegments_axial :')
print(segments_axial)


# Extracting data for time = 0 :

# In[338]:


ar = np.array([[-1.46196869,-1.44590192,-1.44289429,-1.43909605,-1.43902328,-1.4498733,-1.44802323,-1.42451698,-1.33091879,-1.31150646,-1.29821302,-1.46432268,-1.5732341,-1.5736951,-1.5784913,-1.39155897,-1.40134176,-1.40981623,-1.47005831,-1.53196249,-1.52147113,-1.49790626,-1.51765804],
[4.74019817,4.71669717,4.71901424,4.72003391,4.72765532,4.7545356,4.79399716,4.75475943,4.8302209,4.80059702,4.78317192,4.72733644,4.67276963,4.66464892,4.65247688,4.77532363,4.74974287,4.72344367,4.87605296,4.70446165,4.75367219,4.74819702,4.90178458],
[0.92699201,1.01652977,1.10895436,1.19338542,1.27746879,1.3886864,1.48152368,1.33837237,1.31232968,1.04565219,0.82890768,1.34112352,1.32398888,1.05509626,0.83764134,0.92487603,0.46482252,0.12182995,0.04313222,0.93135087,0.47323962,0.12998758,0.04081183]])


# In[339]:


df = pd.DataFrame(ar, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df


# Visualizing data for time = 0 :

# In[340]:


fig1 = plt.figure(figsize=(40,40))
ax1 = plt.axes(projection='3d')


# Drawing the skeleton with the names of the sensors :

# In[341]:


plot_all_sensors_with_names (df,ax1)


# Draw the head :

# In[342]:


x_head = -1.448023
y_head = 4.793997
z_head = 1.481524
ax1.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")


# In[343]:


fig1


# In[344]:


plot_all_segments(df,ax1)
ax1.view_init(0, -35)
fig1


# Change viewing angle :

# In[345]:


ax1.quiver(0,0,0,0,0.0000001,0,length=1.0)
fig1


# Drawing the skeleton with the numbers of sensors :

# In[346]:


fig2 = plt.figure(figsize=(40,40))
ax2 = plt.axes(projection='3d')
plot_all_sensors_with_numbers(df,ax2)


# In[347]:


x_head = -1.448023
y_head = 4.793997
z_head = 1.481524
ax2.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig2


# In[348]:


plot_all_segments(df,ax2)
ax2.view_init(0, -35)
ax2.quiver(0,0,0,0,0.0000001,0,length=1.0)
fig2


# In[ ]:




