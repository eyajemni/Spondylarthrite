#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[74]:


def new_figure () :
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(50,80)
    ax.set_ylim3d(80, 180)
    ax.set_zlim3d(0, 180) 
    ax.view_init(0, -35)
    #ax.view_init(50, -35)
    #ax.set_axis_off()
    return   fig, ax

def data_to_df (data, temps) :
    l=data.shape[1]-5
    datax=[]; datay=[]; dataz=[]
    for i in range (0,l,3):
        datax.append(data.loc[:,data.columns[i]][temps])  #extract all the x's from the first column and store them in datax list
        datay.append(data.loc[:,data.columns[i+1]][temps])  #extract all the y's from the first column and store them in datay list
        dataz.append(data.loc[:,data.columns[i+2]][temps])  #extract all the z's from the first column and store them in dataz list
    ar=np.array([datax,datay,dataz])
    df = pd.DataFrame(ar, index = ['x', 'y', 'z'])
    return df

def plot_all_sensors_with_names (data,ax):
    nb_sensors = len(data.columns)-5
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (sensor_name), size=20, zorder=1,)

def plot_all_sensors_with_numbers (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        z=data.loc[:,sensor_name][1]
        y=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=20, marker='d', color="g")
        ax.text(x,y,z, '%s''%s' % ('  ',i+1), size=5, zorder=1,)

def muscle_groups (data,ax):
    for i in range (51,54):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=20, marker='d', color="g")
        ax.text(x,y,z, '%s''%s' % ('  ',i+1), size=5, zorder=1,)

def plot_all_segments(data,ax) :
    for i in range (0,len(segments_right)) :
        ax.plot(data.loc[:,segments_right[i]].loc['x',:],data.loc[:,segments_right[i]].loc['z',:],data.loc[:,segments_right[i]].loc['y',:], color="b")
    for i in range (0,len(segments_left)) :
        ax.plot(data.loc[:,segments_left[i]].loc['x',:],data.loc[:,segments_left[i]].loc['z',:],data.loc[:,segments_left[i]].loc['y',:], color="r")
    for i in range (0,len(segments_axial)) :
        ax.plot(data.loc[:,segments_axial[i]].loc['x',:],data.loc[:,segments_axial[i]].loc['z',:],data.loc[:,segments_axial[i]].loc['y',:], color="k")



def plot_head (data,ax,temps) :
    x_head = data.loc[:,12][temps]
    y_head = data.loc[:,14][temps]
    z_head = data.loc[:,13][temps]
    ax.scatter(x_head,y_head,z_head,s=400, marker='o', color="k")



def animate(i):
    print(i)
    df= data_to_df (data, i*2)
    ax=new_figure()
    plot_all_sensors_with_numbers(df,ax)
    plot_all_segments(df,ax)
    ax.view_init(10,5)
    plot_head (data,ax,i*2)
    #


# In[61]:


data = pd.read_csv ('P33__0_650.txt', delimiter="\t" , header=None)
print(data)


# In[62]:


data.shape


# In[63]:


dt= pd.DataFrame (data)
dt


# In[64]:


data = pd.read_csv ('P33__0_650.txt', delimiter=",", header=None)
data


# In[65]:


data = data.transpose()
data


# In[66]:


segments_right=np.array([[3,5],[5,6],[6,7],[0,11],[11,12],[12,13]])
segments_left=np.array([[3,8],[8,9],[9,10],[0,14],[14,15],[15,16]])
segments_axial=np.array([[4, 2],[2,3],[3,1],[1,0]])


# In[67]:


print('segments_right :')
print(segments_right)
print('\nsegments_left :')
print(segments_left)
print('\nsegments_axial :')
print(segments_axial)


# In[75]:


l=data.shape[0]
for i in range (10) :
    df= data_to_df (data, i)
    fig, ax = new_figure()
    plot_all_sensors_with_numbers(df,ax)
    plot_all_segments(df,ax)
    #ax.view_init(0,20)
    plot_head (data,ax,i)
    #s= str (i)
    #fig.savefig(s,dpi=500)


# In[76]:



writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)


fig = plt.figure(figsize=(10,10))


ani = matplotlib.animation.FuncAnimation(fig, animate, frames=int(data.shape[0]/10), repeat=True)


# In[ ]:




