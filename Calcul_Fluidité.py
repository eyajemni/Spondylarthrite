#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:/Users/Eya/Desktop/My codes')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import butter, filtfilt, freqz, buttord

from smoothness_2 import sparc
from smoothness_2 import dimensionless_jerk2
from smoothness import spectral_arclength
from smoothness import dimensionless_jerk
from for_paper import plot_different_tasks
from mouvements import data_span

from pylab import *
from scipy import misc


# In[2]:


def Fluidité (file) :
    #Création du nom des colonnes en une liste
    Frame,Pelvis_x,Pelvis_y,Pelvis_z,L5_x,L5_y,L5_z,L3_x,L3_y,L3_z,T12_x,T12_y,T12_z,T8_x,T8_y,T8_z,Neck_x,Neck_y,Neck_z,Head_x,Head_y,Head_z,R_Shou_x,R_Shou_y,R_Shou_z,R_U_Arm_x,R_U_Arm_y,R_U_Arm_z,R_F_Arm_x,R_F_Arm_y,R_F_Arm_z,R_Hand_x,R_Hand_y,R_Hand_z,L_Shou_x,L_Shou_y,L_Shou_z,L_U_Arm_x,L_U_Arm_y,L_U_Arm_z,L_F_Arm_x,L_F_Arm_y,L_F_Arm_z,L_Hand_x,L_Hand_y,L_Hand_z,R_U_Leg_x,R_U_Leg_y,R_U_Leg_z,R_L_Leg_x,R_L_Leg_y,R_L_Leg_z,R_Foot_x,R_Foot_y,R_Foot_z,R_Toe_x,R_Toe_y,R_Toe_z,L_U_Leg_x,L_U_Leg_y,L_U_Leg_z,L_L_Leg_x,L_L_Leg_y,L_L_Leg_z,L_Foot_x,L_Foot_y,L_Foot_z,L_Toe_x,L_Toe_y,L_Toe_z = ([] for i in range (70))
                       
    L =[Frame,Pelvis_x,Pelvis_y,Pelvis_z,L5_x,L5_y,L5_z,L3_x,L3_y,L3_z,T12_x,T12_y,T12_z,T8_x,T8_y,T8_z,Neck_x,Neck_y,Neck_z,Head_x,Head_y,Head_z,R_Shou_x,R_Shou_y,R_Shou_z,R_U_Arm_x,R_U_Arm_y,R_U_Arm_z,R_F_Arm_x,R_F_Arm_y,R_F_Arm_z,R_Hand_x,R_Hand_y,R_Hand_z,L_Shou_x,L_Shou_y,L_Shou_z,L_U_Arm_x,L_U_Arm_y,L_U_Arm_z,L_F_Arm_x,L_F_Arm_y,L_F_Arm_z,L_Hand_x,L_Hand_y,L_Hand_z,R_U_Leg_x,R_U_Leg_y,R_U_Leg_z,R_L_Leg_x,R_L_Leg_y,R_L_Leg_z,R_Foot_x,R_Foot_y,R_Foot_z,R_Toe_x,R_Toe_y,R_Toe_z,L_U_Leg_x,L_U_Leg_y,L_U_Leg_z,L_L_Leg_x,L_L_Leg_y,L_L_Leg_z,L_Foot_x,L_Foot_y,L_Foot_z,L_Toe_x,L_Toe_y,L_Toe_z]



    for i in range (70):
        L[i]=file.loc[:,file.columns[i]]
    for i in range (len(L)) :
        L[i]= np.array (L[i])
    


    #Calculation of the spectral arc length
    Fluidité = 0 
    fs = 60

    for i in range (1,24):
        SAL_x = spectral_arclength(L[i], fs)[0] 
        SAL_y = spectral_arclength(L[i+1], fs)[0]
        SAL_z = spectral_arclength(L[i+2], fs)[0]
        F = ( SAL_x**2 + SAL_y**2 + SAL_z**2 ) /23
        Fluidité += F
        
    return Fluidité


# In[3]:


file =pd.read_excel('C:/Users/Eya/Desktop/Xsens/LOMBALGIE/SUJETS_SAINS/thierry/thierry-002.xlsx', sheet_name ="Segment Position")

print (Fluidité(file))

