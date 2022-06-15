#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[2]:


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
         ax.scatter(x,y,z,s=200, marker='d', color="g")   
    
    def plot_marker_with_name (self,ax):
        ax.scatter(x,y,z,s=200, marker='d', color="g")
        ax.text(x,y,z,'%s' % (self.name), size=20, zorder=1,)

    def plot_marker_with_number (self,ax):
            ax.scatter(x,y,z,s=20, marker='d', color="g")
            ax.text(x,y,z, '%s''%s' % ('  ',self.number), size=5, zorder=1,)


# In[3]:


class Link: 
  
    def __init__(self, marker1, marker2, color):      
        self.marker1 = marker1
        self.marker2 = marker2
        self.color = color
        
        
    def plot_link (self,ax):
        X= [marker1.x , marker2.x]
        Y= [marker1.y , marker2.y]
        Z= [marker1.z , marker2.z]
        ax.plot(X,Y,Z, color=self.color)

        


# In[21]:


class Skeleton: 
    markers = []
    links = []
    
    def __init__(self, markers, links):      
        self.markers = markers
        self.links = links

    def plot_all_markers(self): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            marker= markers[i]
            marker.plot_marker()
     
    
    def plot_all_markers_with_names(self): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            marker= markers[i]
            marker.plot_marker_with_name()
                                              
    
    def plot_all_markers_with_numbers(self): 
        nMarkers = len(markers)
        for i in range (0,nMarkers):
            marker= markers[i]
            marker.plot_marker_with_numbers()
    
    
    def plot_all_links(self) :
        nLinks = len(links)
        for i in range (0,nLinks) :
            link = links[i]
            link.plot_link()
        
        
 


# In[18]:


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
    


# In[6]:


data = pd.read_csv ('P33__0_650.txt', delimiter="\t" , header=None)
print(data)


# In[7]:


dt= pd.DataFrame (data)
dt


# In[8]:


data = pd.read_csv ('P33__0_650.txt', delimiter=",", header=None)
data


# In[9]:


data = data.transpose()
data


# In[23]:


l=data.shape[0]

links_right=np.array([[3,5],[5,6],[6,7],[0,11],[11,12],[12,13]])
links_left=np.array([[3,8],[8,9],[9,10],[0,14],[14,15],[15,16]])
links_axial=np.array([[4, 2],[2,3],[3,1],[1,0]])
links=[links_right,links_left,links_axial]

for i in range (10) :
    df= data_to_df (data, i)
    nMarkers = len(df.columns)
    fig, ax = new_figure()
    for i in range (0,nMarkers):
        markerName = df.columns[i]
        x=df.loc[:,markerName][0]
        z=df.loc[:,markerName][1]
        y=df.loc[:,markerName][2]
        marker= Marker (x,y,z,markerName,i+1)
        marker.plot_marker_with_number(ax) 
        
        
        
    for n in range (3) :
        links_=links[n]
        for k in range (0,len(links_)) :
            xx= np.array (df.loc[:,links_[k]].loc['x',:])
            x1= xx[0]; x2 =xx[1]
            yy=np.array(df.loc[:,links_[k]].loc['z',:])
            y1= yy[0]; y2 = yy[1]
            zz= np.array (df.loc[:,links_[k]].loc['y',:])
            z1= zz[0]; z2 = zz[1]
            number1 = links_[k][0] ; number2 = links_[k][1]
            name1 = df.columns[number1] ; name2 = df.columns[number2]

            marker1= Marker (x1,y1,z1,name1, number1)
            marker2= Marker (x2,y2,z2,name2, number2)
            
            if n==0 : link= Link(marker1,marker2,'b')
            if n==1 : link= Link(marker1,marker2,'r')
            if n==2 : link= Link(marker1,marker2,'k')

            
            link.plot_link(ax)
 
    #plot_head (df,ax,i)
    


# In[ ]:



writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)


fig = plt.figure(figsize=(10,10))


ani = matplotlib.animation.FuncAnimation(fig, animate, frames=int(data.shape[0]/10), repeat=True)

