#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries :

# In[303]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio


# ## Functions Defenition :

# In[256]:


def new_figure () :
    fig = plt.figure(figsize=(40,40))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X', size=40)
    ax.set_ylabel('Y', size=40)
    ax.set_zlabel('Z', size=40)
    ax.set_xlim3d(-2, 1.2)
    ax.set_ylim3d(1, 7)
    ax.set_zlim3d(0, 1.7)
    return fig, ax


# In[257]:


def plot_all_sensors_with_names (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (sensor_name), size=20, zorder=1,)


# In[258]:


def plot_all_sensors_with_numbers (data,ax):
    nb_sensors = len(data.columns)
    for i in range (0,nb_sensors):
        sensor_name = data.columns[i]
        x=data.loc[:,sensor_name][0]
        y=data.loc[:,sensor_name][1]
        z=data.loc[:,sensor_name][2]
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z, '%s''%s' % ('  ',i+1), size=15, zorder=1,)


# In[259]:


def plot_all_segments(data,ax) :
    for i in range (0,len(segments_right)) :
        ax.plot(data.loc[:,segments_right[i]].loc['x',:],data.loc[:,segments_right[i]].loc['y',:],data.loc[:,segments_right[i]].loc['z',:], color="b")
    for i in range (0,len(segments_left)) :
        ax.plot(data.loc[:,segments_left[i]].loc['x',:],data.loc[:,segments_left[i]].loc['y',:],data.loc[:,segments_left[i]].loc['z',:], color="r")
    for i in range (0,len(segments_axial)) :
        ax.plot(data.loc[:,segments_axial[i]].loc['x',:],data.loc[:,segments_axial[i]].loc['y',:],data.loc[:,segments_axial[i]].loc['z',:], color="k")


# In[260]:


def plot_head (data,ax,temps) :
    x_head = data.loc[:,'Head x'][temps]
    y_head = data.loc[:,'Head y'][temps]
    z_head = data.loc[:,'Head z'][temps]
    ax.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")


# # Reading movement stairs data :

# In[261]:


data=pd.read_excel('Pilot_006_JeEy_Stairs.xlsx', sheet_name ="Segment Position")
data


# In[262]:


columns= data.columns
columns


# ## Preparing 3 arrays : 
# 1. segments_right contains the segments of the right limbs 
# 2. segments_left contains the segments of the left limbs 
# 3. segments_axial contains the segments of the axial part

# In[263]:


segments_right=np.array([['Neck', 'Right Shoulder'],['Right Upper Arm', 'Right Shoulder'],['Right Forearm','Right Hand'],['Right Forearm','Right Upper Arm'],['Right Upper Leg', 'Right Lower Leg'],['Right Upper Leg', 'Pelvis'],['Right Foot', 'Right Lower Leg'],['Right Foot', 'Right Toe']])
segments_left=np.array([['Left Forearm','Left Hand'],['Left Forearm','Left Upper Arm'],['Neck', 'Left Shoulder'],['Left Upper Arm', 'Left Shoulder'],['Left Upper Leg', 'Pelvis'],['Left Foot', 'Left Lower Leg'],['Left Foot', 'Left Toe'],['Left Upper Leg', 'Left Lower Leg']])
segments_axial=np.array([['Neck', 'Head'],['T8', 'Neck'],['T12', 'T8'],['L3', 'T12'],['L3', 'L5'],['L5', 'Pelvis']])


# In[264]:


print('segments_right :')
print(segments_right)
print('\nsegments_left :')
print(segments_left)
print('\nsegments_axial :')
print(segments_axial)


# # Extracting data for time = 0 :

# In[265]:


l=data.shape[1]
datax=[]; datay=[]; dataz=[]
for i in range (1,l,3):
    datax.append(data.loc[:,data.columns[i]][0])  #extract all the x's from the first column and store them in datax list
    datay.append(data.loc[:,data.columns[i+1]][0])  #extract all the y's from the first column and store them in datay list
    dataz.append(data.loc[:,data.columns[i+2]][0])  #extract all the z's from the first column and store them in dataz list


# In[266]:


ar=np.array([datax,datay,dataz])
ar


# In[267]:


df = pd.DataFrame(ar, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df


# In[268]:


new_df = df.transpose()
new_df


# In[269]:


new_df.describe()


# # Visualizing data for time = 0 :

# ## Drawing the skeleton with the names of the sensors :

# In[270]:


fig1, ax1 = new_figure()
plot_all_sensors_with_names (df,ax1)
plot_all_segments(df,ax1)
ax1.view_init(0, -35)


# #### Draw the head :

# In[271]:


x_head = data.loc[:,'Head x'][0]
y_head = data.loc[:,'Head y'][0]
z_head = data.loc[:,'Head z'][0]
ax1.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig1


# ### Saving the image

# In[308]:


images = []
ax1.azim = ax1.azim+1.1
fig1.canvas.draw()
image1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8') 
images.append(image1) 
imageio.mimsave('test.gif', images)


# ## Drawing the skeleton with the numbers of sensors :

# In[272]:


fig2, ax2 = new_figure()
plot_all_sensors_with_numbers(df,ax2)
plot_all_segments(df,ax2)
ax2.view_init(0, -35)
ax2.quiver(0,0,0,0,0.0000001,0,length=1.0)


# #### Draw the head

# In[273]:


x_head = data.loc[:,'Head x'][0]
y_head = data.loc[:,'Head y'][0]
z_head = data.loc[:,'Head z'][0]
ax2.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig2


# # Extracting data for time = 1000 :

# In[274]:


l=data.shape[1]
datax=[]; datay=[]; dataz=[]
for i in range (1,l,3):
    datax.append(data.loc[:,data.columns[i]][1000])  #extract all the x's from the first column and store them in datax list
    datay.append(data.loc[:,data.columns[i+1]][1000])  #extract all the y's from the first column and store them in datay list
    dataz.append(data.loc[:,data.columns[i+2]][1000])  #extract all the z's from the first column and store them in dataz list


# In[275]:


ar2=np.array([datax,datay,dataz])
ar2


# In[276]:


df2 = pd.DataFrame(ar2, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df2


# In[277]:


new_df2 = df2.transpose()
new_df2


# In[278]:


new_df2.describe()


# # Visualizing data for time = 1000 :

# ## Drawing the skeleton with the names of the sensors :

# In[279]:


fig3, ax3 = new_figure()
plot_all_sensors_with_names (df2,ax3)
plot_all_segments(df2,ax3)
ax3.view_init(0, -35)
ax3.quiver(0,0,0,0,0.0000001,0,length=1.0)


# In[280]:


x_head = data.loc[:,'Head x'][1000]
y_head = data.loc[:,'Head y'][1000]
z_head = data.loc[:,'Head z'][1000]
ax3.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig3


# ## Drawing the skeleton with the numbers of sensors :

# In[281]:


fig4, ax4 = new_figure()
plot_all_sensors_with_numbers(df2,ax4)
plot_all_segments(df2,ax4)
ax4.view_init(0, -35)
ax4.quiver(0,0,0,0,0.0000001,0,length=1.0)


# In[282]:


x_head = data.loc[:,'Head x'][1000]
y_head = data.loc[:,'Head y'][1000]
z_head = data.loc[:,'Head z'][1000]
ax4.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig4


# # Extracting data for time = 2500 :

# In[283]:


l=data.shape[1]
datax=[]; datay=[]; dataz=[]
for i in range (1,l,3):
    datax.append(data.loc[:,data.columns[i]][2500])  #extract all the x's from the first column and store them in datax list
    datay.append(data.loc[:,data.columns[i+1]][2500])  #extract all the y's from the first column and store them in datay list
    dataz.append(data.loc[:,data.columns[i+2]][2500])  #extract all the z's from the first column and store them in dataz list


# In[284]:


ar3=np.array([datax,datay,dataz])
ar3


# In[285]:


df3 = pd.DataFrame(ar3, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df3


# In[286]:


new_df3 = df3.transpose()
new_df3


# In[287]:


new_df3.describe()


# # Visualizing data for time = 2500 :

# ## Drawing the skeleton with the names of the sensors :

# In[288]:


fig5, ax5 = new_figure()
plot_all_sensors_with_names (df3,ax5)
plot_all_segments(df3,ax5)
ax5.view_init(0, -35)
ax5.quiver(0,0,0,0,0.0000001,0,length=1.0)


# In[289]:


x_head = data.loc[:,'Head x'][2500]
y_head = data.loc[:,'Head y'][2500]
z_head = data.loc[:,'Head z'][2500]
ax5.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig5


# ## Drawing the skeleton with the numbers of sensors :

# In[290]:


fig6, ax6 = new_figure()
plot_all_sensors_with_numbers(df3,ax6)
plot_all_segments(df3,ax6)
ax6.view_init(0, -35)
ax6.quiver(0,0,0,0,0.0000001,0,length=1.0)


# In[291]:


x_head = data.loc[:,'Head x'][2500]
y_head = data.loc[:,'Head y'][2500]
z_head = data.loc[:,'Head z'][2500]
ax6.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig6


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Extracting data for time = 4653 :

# In[292]:


l=data.shape[1]
datax=[]; datay=[]; dataz=[]
for i in range (1,l,3):
    datax.append(data.loc[:,data.columns[i]][4653])  #extract all the x's from the first column and store them in datax list
    datay.append(data.loc[:,data.columns[i+1]][4653])  #extract all the y's from the first column and store them in datay list
    dataz.append(data.loc[:,data.columns[i+2]][4653])  #extract all the z's from the first column and store them in dataz list


# In[293]:


ar4=np.array([datax,datay,dataz])
ar4


# In[294]:


df4 = pd.DataFrame(ar4, index = ['x', 'y', 'z'], columns = ['Pelvis','L5','L3','T12','T8','Neck','Head','Right Shoulder','Right Upper Arm','Right Forearm','Right Hand','Left Shoulder','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Right Toe','Left Upper Leg','Left Lower Leg','Left Foot','Left Toe'])
df4


# In[295]:


new_df4 = df4.transpose()
new_df4


# In[296]:


new_df4.describe()


# In[ ]:





# ## Drawing the skeleton with the names of the sensors :

# In[297]:


fig7, ax7 = new_figure()
plot_all_sensors_with_names (df4,ax7)
plot_all_segments(df4,ax7)
ax7.view_init(0, -35)


# In[298]:


x_head = data.loc[:,'Head x'][4653]
y_head = data.loc[:,'Head y'][4653]
z_head = data.loc[:,'Head z'][4653]
ax7.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig7


# ## Drawing the skeleton with the numbers of sensors :

# In[299]:


fig8, ax8 = new_figure()
plot_all_sensors_with_numbers(df4,ax8)
plot_all_segments(df4,ax8)
ax8.view_init(0, -35)


# In[300]:


x_head = data.loc[:,'Head x'][4653]
y_head = data.loc[:,'Head y'][4653]
z_head = data.loc[:,'Head z'][4653]
ax8.scatter(x_head,y_head,z_head,s=2500, marker='o', color="k")
fig8


# In[ ]:





# In[ ]:


images = []
ax.azim = ax.azim+1.1
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8') 
        images.append(image.reshape(300, 500, 3)) 
        imageio.mimsave('test.gif', images)

