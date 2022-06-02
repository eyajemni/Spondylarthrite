# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:30:19 2022

@author: imoussaten
"""


# import numpy as np
# from matplotlib import pyplot as plt
# from celluloid import Camera

# fig, axes = plt.subplots(2)
# camera = Camera(fig)
# t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
# for i in t:
#     axes[0].plot(t, np.sin(t + i), color='blue')
#     axes[1].plot(t, np.sin(t - i), color='blue')
#     camera.snap()
    

# animation = camera.animate()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


overdoses = pd.read_excel('overdose_data_1999-2015.xlsx',sheet_name='Online',skiprows =6)

def get_data(table,rownum,title):
    data = pd.DataFrame(table.loc[rownum][2:]).astype(float)
    data.columns = {title}
    return data


#%matplotlib notebook
title = 'Heroin Overdoses'
d = get_data(overdoses,18,title)
x = np.array(d.index)
y = np.array(d['Heroin Overdoses'])
overdose = pd.DataFrame(y,x)
#XN,YN = augment(x,y,10)
#augmented = pd.DataFrame(YN,XN)
overdose.columns = {title}


Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(10,6))
plt.xlim(1999, 2016)
plt.ylim(np.min(overdose)[0], np.max(overdose)[0])
plt.xlabel('Year',fontsize=20)
plt.ylabel(title,fontsize=20)
plt.title('Heroin Overdoses per Year',fontsize=20)

def animate(i):
    data = overdose.iloc[:int(i+1)] #select data range
    p = sns.lineplot(x=data.index, y=data[title], data=data, color="r")
    p.tick_params(labelsize=17)
    plt.setp(p.lines,linewidth=7)
    
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=17, repeat=True)


#ani.save('HeroinOverdosesJumpy.mp4', writer=writer)