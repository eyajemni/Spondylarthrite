#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[76]:


class Marker:
    x = 0
    y = 0
    z = 0 
    name = ''
    number =0
    
    def __init__(self, x,y,z,name,number):      
        self.x = x
        self.y = y
        self.z = z
        self.name = name
        self.number = number
    
    def plot_marker (self, ax):
        ax.scatter(x,y,z,s=50, marker='d', color="g")   
    
    def plot_marker_with_name (self,ax):
        self.plot_marker (ax)
        ax.text(x,y,z,'%s' % (self.name), size=50, zorder=1,)

    def plot_marker_with_number (self,ax):
        self.plot_marker (ax)
        ax.text(x,y,z, '%s''%s' % ('  ',self.number), size=50, zorder=1,)


# In[77]:


marker01= Marker(0,0,0,'',0)
marker02= Marker(0,0,0,'',0)

class Link: 
    marker1 = marker01
    marker2 = marker02
    color = 'k'
  
    def __init__(self, marker1, marker2, color):      
        self.marker1 = marker1
        self.marker2 = marker2
        self.color = color
        
        
    def plot_link (self,ax):
        X= [marker1.x , marker2.x]
        Y= [marker1.y , marker2.y]
        Z= [marker1.z , marker2.z]
        ax.plot(X,Y,Z, color=self.color)

        


# In[78]:


class Skeleton: 
    markers = []
    links = []
    
    def __init__(self, markers, links):      
        self.markers = markers
        self.links = links

    def plot_all_markers(self, ax): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            markers[i].plot_marker(ax)
     
    
    def plot_all_markers_with_names(self, ax): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            markers[i].plot_marker_with_name(ax)
                                              
    
    def plot_all_markers_with_numbers(self, ax): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            markers[i].plot_marker_with_number(ax)
    
    
    def plot_all_links(self, ax) :
        nLinks = len(links)
        for i in range (0,nLinks) :
            links[i].plot_link(ax)
        
        
 


# In[79]:


def new_figure () :
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    #ax.set_xlim3d(50,80)
    #ax.set_ylim3d(80, 180)
    #ax.set_zlim3d(0, 180) 
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
    


# In[80]:


data = pd.read_csv ('P33__0_637.txt', delimiter="\t" , header=None)
print(data)


# In[81]:


dt= pd.DataFrame (data)
dt


# In[82]:


data = pd.read_csv ('P33__0_637.txt', delimiter=",", header=None)
data


# In[83]:


data = data.transpose()
data


# In[84]:


d=data.loc[:,0]


# In[85]:


l=data.shape[0]
links_right=np.array([[3,5],[5,6],[6,7],[0,11],[11,12],[12,13]])
links_left=np.array([[3,8],[8,9],[9,10],[0,14],[14,15],[15,16]])
links_axial=np.array([[4, 2],[2,3],[3,1],[1,0]])
links_num=[links_right,links_left,links_axial]
markers_name=['Pelvis','Neck','Shoulders','Head','Right Upper Arm','Right Forearm','Right Hand','Left Upper Arm','Left Forearm','Left Hand','Right Upper Leg','Right Lower Leg','Right Foot','Left Upper Leg','Left Lower Leg','Left Foot']


# In[86]:


links_right


# In[93]:


markers=[]
links=[]


for i in range (10) :
    fig, ax = new_figure()
    df= data_to_df (data, i)
    
    
    #construire la liste des markers
    nMarkers = len(df.columns)
    for i in range (0,nMarkers): 
        markerNum = df.columns[i]
        x=df.loc[:,markerNum][0]
        z=df.loc[:,markerNum][1]
        y=df.loc[:,markerNum][2]
        marker = Marker (x,y,z,markers_name[i-1],i+1)
        markers.append(marker)
        markers[i].plot_marker(ax)


        
    #construire la liste des links       
    for n in range (3) :
        links_=links_num[n]
        for k in range (0,len(links_)) :
            xx= np.array (df.loc[:,links_[k]].loc['x',:])
            x1= xx[0]; x2 =xx[1]
            yy=np.array(df.loc[:,links_[k]].loc['z',:])
            y1= yy[0]; y2 = yy[1]
            zz= np.array (df.loc[:,links_[k]].loc['y',:])
            z1= zz[0]; z2 = zz[1]
            number1 = links_[k][0]-1 ; number2 = links_[k][1]-1
            name1 = markers_name[number1] ; name2 = markers_name[number2]

            marker1= Marker (x1,y1,z1,name1, number1)
            marker2= Marker (x2,y2,z2,name2, number2)

            if n==0 : 
                link= Link(marker1,marker2,'b')
                links.append(link)
                link.plot_link(ax)
            if n==1 : 
                link= Link(marker1,marker2,'r')
                links.append(link)
                link.plot_link(ax)
            if n==2 : 
                link= Link(marker1,marker2,'k')
                links.append(link)
                link.plot_link(ax)
                


# In[88]:


fig=plt.figure()
dd=np.array([m for m in range (0,180)])
plt.plot(dd,data.loc[:,0])
plt.title('Pelvis x au cours du temps')


# In[89]:


fig=plt.figure()
dd=np.array([m for m in range (0,180)])
plt.plot(dd,data.loc[:,1])
plt.title('Pelvis y au cours du temps')


# In[90]:


fig=plt.figure()
dd=np.array([m for m in range (0,180)])
plt.plot(dd,data.loc[:,2])
plt.title('Pelvis z au cours du temps')


# In[91]:


skeleton = Skeleton (markers, links)

fig, ax = new_figure()

skeleton.plot_all_markers(ax)


# In[92]:



writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)


fig = plt.figure(figsize=(10,10))


ani = matplotlib.animation.FuncAnimation(fig, animate, frames=int(data.shape[0]/10), repeat=True)

