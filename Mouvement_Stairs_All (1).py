#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries :

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Functions Defenition :

# In[60]:


def new_figure () :
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.set_facecolor((0, 0, 0))
    #ax.set_xlabel('X', size=40)
    #ax.set_ylabel('Y', size=40)
    #ax.set_zlabel('Z', size=40)
    ax.set_xlim3d(-2, 1.2)
    ax.set_ylim3d(1, 7)
    ax.set_zlim3d(0, 1.7)
    # ax.set_xticks=[]
    # ax.set_yticks=[]
    # ax.set_zticks=[]
    #ax.set_visible=False
    ax.set_axis_off()
    return fig, ax



# In[61]:


def data_to_df (data, temps) :
    l=data.shape[1]
    datax=[]; datay=[]; dataz=[]
    for i in range (1,l,3):
        datax.append(data.loc[:,data.columns[i]][temps])  #extract all the x's from the first column and store them in datax list
        datay.append(data.loc[:,data.columns[i+1]][temps])  #extract all the y's from the first column and store them in datay list
        dataz.append(data.loc[:,data.columns[i+2]][temps])  #extract all the z's from the first column and store them in dataz list
    ar=np.array([datax,datay,dataz])
    df = pd.DataFrame(ar, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
    return df


# In[62]:


def plot_all_sensors_with_names (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (sensor_name), size=20, zorder=1,)


# In[63]:


def plot_all_sensors_with_numbers (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=50, marker='d', color="g")
        #ax.text(x,y,z, '%s''%s' % ('  ',i+1), size=15, zorder=1,)


# In[64]:


def plot_all_segments(data,ax) :
    for i in range (0,len(segments_right)) :
        ax.plot(data.loc[:,segments_right[i]].loc['x',:],data.loc[:,segments_right[i]].loc['y',:],data.loc[:,segments_right[i]].loc['z',:], color="w")
    for i in range (0,len(segments_left)) :
        ax.plot(data.loc[:,segments_left[i]].loc['x',:],data.loc[:,segments_left[i]].loc['y',:],data.loc[:,segments_left[i]].loc['z',:], color="r")
    for i in range (0,len(segments_axial)) :
        ax.plot(data.loc[:,segments_axial[i]].loc['x',:],data.loc[:,segments_axial[i]].loc['y',:],data.loc[:,segments_axial[i]].loc['z',:], color="k")


# In[65]:


def plot_head (data,ax,temps) :
    x_head = data.loc[:,'Head x'][temps]
    y_head = data.loc[:,'Head y'][temps]
    z_head = data.loc[:,'Head z'][temps]
    ax.scatter(x_head,y_head,z_head,s=200, marker='o', color="r")


# # Reading movement stairs data :

# In[66]:


data=pd.read_excel('Pilot_006_JeEy_Stairs.xlsx', sheet_name ="Segment Position")
data


# In[67]:


columns= data.columns
columns


# ## Preparing 3 arrays : 
# 1. segments_right contains the segments of the right limbs 
# 2. segments_left contains the segments of the left limbs 
# 3. segments_axial contains the segments of the axial part

# In[68]:


segments_right=np.array([['Neck', 'Right Shoulder'],['Right Upper Arm', 'Right Shoulder'],['Right Forearm','Right Hand'],['Right Forearm','Right Upper Arm'],['Right Upper Leg', 'Right Lower Leg'],['Right Upper Leg', 'Pelvis'],['Right Foot', 'Right Lower Leg'],['Right Foot', 'Right Toe']])
segments_left=np.array([['Left Forearm','Left Hand'],['Left Forearm','Left Upper Arm'],['Neck', 'Left Shoulder'],['Left Upper Arm', 'Left Shoulder'],['Left Upper Leg', 'Pelvis'],['Left Foot', 'Left Lower Leg'],['Left Foot', 'Left Toe'],['Left Upper Leg', 'Left Lower Leg']])
segments_axial=np.array([['Neck', 'Head'],['T8', 'Neck'],['T12', 'T8'],['L3', 'T12'],['L3', 'L5'],['L5', 'Pelvis']])


# In[69]:


print('segments_right :')
print(segments_right)
print('\nsegments_left :')
print(segments_left)
print('\nsegments_axial :')
print(segments_axial)


# # Visualising movement stairs data :

# In[70]:


l=data.shape[0]
for i in range ( int(l/20) ) :
    df= data_to_df (data, i*20)
    fig, ax = new_figure()
    plot_all_sensors_with_numbers(df,ax)
    plot_all_segments(df,ax)
    ax.view_init(0, -35)
    plot_head (data,ax,i)
    # ax.set_xlim3d([np.min(df.x), np.max(df.x)])
    # ax.set_ylim3d([np.min(df.y), np.max(df.y)])
    # ax.set_zlim3d([np.min(df.z), np.max(df.z)])
    s= 'IMG/'+str (i)
    fig.savefig(s,dpi=500)

