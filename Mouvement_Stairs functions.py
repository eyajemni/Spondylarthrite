#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Functions Defenition :

# In[108]:


def plot_all_sensors (data, tab):
    for i in range (0,23):
        x=tab[0][i]
        y=tab[1][i]
        z=tab[2][i]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (data.columns[i]), size=20, zorder=1,)


# In[109]:


def segment(data) :
    ax.plot(data.loc[:,['Left Forearm','Left Hand']].loc['x',:],data.loc[:,['Left Forearm','Left Hand']].loc['y',:],data.loc[:,['Left Forearm','Left Hand']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Left Forearm','Left Upper Arm']].loc['x',:],data.loc[:,['Left Forearm','Left Upper Arm']].loc['y',:],data.loc[:,['Left Forearm','Left Upper Arm']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Neck', 'Head']].loc['x',:],df.loc[:,['Neck', 'Head']].loc['y',:],df.loc[:,['Neck', 'Head']].loc['z',:], color="k")
    ax.plot(data.loc[:,['Neck', 'Right Shoulder']].loc['x',:],data.loc[:,['Neck', 'Right Shoulder']].loc['y',:],data.loc[:,['Neck', 'Right Shoulder']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Neck', 'Left Shoulder']].loc['x',:],data.loc[:,['Neck', 'Left Shoulder']].loc['y',:],data.loc[:,['Neck', 'Left Shoulder']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Left Upper Arm', 'Left Shoulder']].loc['x',:],data.loc[:,['Left Upper Arm', 'Left Shoulder']].loc['y',:],data.loc[:,['Left Upper Arm', 'Left Shoulder']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Right Upper Arm', 'Right Shoulder']].loc['x',:],data.loc[:,['Right Upper Arm', 'Right Shoulder']].loc['y',:],data.loc[:,['Right Upper Arm', 'Right Shoulder']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Right Forearm','Right Hand']].loc['x',:],data.loc[:,['Right Forearm','Right Hand']].loc['y',:],data.loc[:,['Right Forearm','Right Hand']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Right Forearm','Right Upper Arm']].loc['x',:],data.loc[:,['Right Forearm','Right Upper Arm']].loc['y',:],data.loc[:,['Right Forearm','Right Upper Arm']].loc['z',:], color="b")
    ax.plot(data.loc[:,['T8', 'Neck']].loc['x',:],data.loc[:,['T8', 'Neck']].loc['y',:],data.loc[:,['T8', 'Neck']].loc['z',:], color="k")
    ax.plot(data.loc[:,['T12', 'T8']].loc['x',:],data.loc[:,['T12', 'T8']].loc['y',:],data.loc[:,['T12', 'T8']].loc['z',:], color="k")
    ax.plot(data.loc[:,['L3', 'T12']].loc['x',:],data.loc[:,['L3', 'T12']].loc['y',:],data.loc[:,['L3', 'T12']].loc['z',:], color="k")
    ax.plot(data.loc[:,['L3', 'L5']].loc['x',:],data.loc[:,['L3', 'L5']].loc['y',:],data.loc[:,['L3', 'L5']].loc['z',:], color="k") 
    ax.plot(data.loc[:,['L5', 'Pelvis']].loc['x',:],data.loc[:,['L5', 'Pelvis']].loc['y',:],data.loc[:,['L5', 'Pelvis']].loc['z',:], color="k")
    ax.plot(data.loc[:,['Right Upper Leg', 'Pelvis']].loc['x',:],data.loc[:,['Right Upper Leg', 'Pelvis']].loc['y',:],data.loc[:,['Right Upper Leg', 'Pelvis']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Left Upper Leg', 'Pelvis']].loc['x',:],data.loc[:,['Left Upper Leg', 'Pelvis']].loc['y',:],data.loc[:,['Left Upper Leg', 'Pelvis']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Left Upper Leg', 'Left Lower Leg']].loc['x',:],data.loc[:,['Left Upper Leg', 'Left Lower Leg']].loc['y',:],data.loc[:,['Left Upper Leg', 'Left Lower Leg']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Right Upper Leg', 'Right Lower Leg']].loc['x',:],data.loc[:,['Right Upper Leg', 'Right Lower Leg']].loc['y',:],data.loc[:,['Right Upper Leg', 'Right Lower Leg']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Right Foot', 'Right Lower Leg']].loc['x',:],data.loc[:,['Right Foot', 'Right Lower Leg']].loc['y',:],data.loc[:,['Right Foot', 'Right Lower Leg']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Left Foot', 'Left Lower Leg']].loc['x',:],data.loc[:,['Left Foot', 'Left Lower Leg']].loc['y',:],data.loc[:,['Left Foot', 'Left Lower Leg']].loc['z',:], color="r")
    ax.plot(data.loc[:,['Right Foot', 'Right Toe']].loc['x',:],data.loc[:,['Right Foot', 'Right Toe']].loc['y',:],data.loc[:,['Right Foot', 'Right Toe']].loc['z',:], color="b")
    ax.plot(data.loc[:,['Left Foot', 'Left Toe']].loc['x',:],data.loc[:,['Left Foot', 'Left Toe']].loc['y',:],data.loc[:,['Left Foot', 'Left Toe']].loc['z',:], color="r")


# Visualisation of movements stairs data

# In[110]:


data=pd.read_excel('Pilot_006_JeEy_Stairs.xlsx', sheet_name ="Segment Position")
data


# In[111]:


columns= data.columns
columns


# In[112]:


ar = np.array([[-1.46196869,-1.44590192,-1.44289429,-1.43909605,-1.43902328,-1.4498733,-1.44802323,-1.42451698,-1.33091879,-1.31150646,-1.29821302,-1.46432268,-1.5732341,-1.5736951,-1.5784913,-1.39155897,-1.40134176,-1.40981623,-1.47005831,-1.53196249,-1.52147113,-1.49790626,-1.51765804],
[4.74019817,4.71669717,4.71901424,4.72003391,4.72765532,4.7545356,4.79399716,4.75475943,4.8302209,4.80059702,4.78317192,4.72733644,4.67276963,4.66464892,4.65247688,4.77532363,4.74974287,4.72344367,4.87605296,4.70446165,4.75367219,4.74819702,4.90178458],
[0.92699201,1.01652977,1.10895436,1.19338542,1.27746879,1.3886864,1.48152368,1.33837237,1.31232968,1.04565219,0.82890768,1.34112352,1.32398888,1.05509626,0.83764134,0.92487603,0.46482252,0.12182995,0.04313222,0.93135087,0.47323962,0.12998758,0.04081183]])


# In[113]:


df = pd.DataFrame(ar, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df


# In[114]:


fig = plt.figure(figsize=(40,40))
ax = plt.axes(projection='3d')


# In[115]:


plot_all_sensors (df,ar)


# In[116]:


fig


# In[117]:


segment(df)
ax.view_init(0, -35)
fig


# In[118]:


ax.quiver(0,0,0,0,0.0000001,0,length=1.0)
fig


# In[119]:


theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(0.005)
x1 = r*np.cos(theta)-1.25
x2 = r*np.sin(theta)+1.6
y1=np.zeros(100)
k=[50]*100
ax.plot(x1, k, x2, color="k")
fig
plt.show()

# In[ ]:




