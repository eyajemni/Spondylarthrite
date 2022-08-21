#!/usr/bin/env python
# coding: utf-8

# # Max
# ## Mouvement : Extension du rachis
# Data : PRE & POST lombalgie \
# ROM : 2 angles

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# In[3]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\PRE')


# In[4]:


VM=[]
AM=[]
VAM=[]
AAM=[]
CMM=[]
VCMM=[]
ACMM=[]
AT1M=[]
AT2M=[]


Files=['LIEURAIN\\LIEURAIN-003.xlsx','BUTSCHER\\BUTSCHER-002.xlsx','ADJADJ\\Extension_001.xlsx']
for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    
    AT1=angles.loc[:,angles.columns[12]] #Angles of T8
    AT2=angles.loc[:,angles.columns[15]] 
    
    
    AT1M.append(np.max(AT1))
    AT2M.append(np.max(AT2))
    
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    mean = np.mean(A)
    AM.append(mean)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    mean = np.mean(VA)
    VAM.append(mean)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    mean = np.mean(AA)
    AAM.append(mean)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    mean = np.mean(CM)
    CMM.append(mean)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    mean = np.mean(VCM)
    VCMM.append(mean)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    mean = np.mean(ACM)
    ACMM.append(mean)
    
    

    


# In[ ]:





# In[5]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\POST')


# In[6]:



Files=['LIEURAIN\\Emmanuel-002.xlsx','BUTSCHER\\BUTSCHER-YANN-002.xlsx','ADJADJA\\ADJADJA NICOLAS-004.xlsx']
for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    
    AT1=angles.loc[:,angles.columns[12]] #Angles of T8
    AT2=angles.loc[:,angles.columns[15]] 
    
    
    AT1M.append(np.mean(AT1))
    AT2M.append(np.mean(AT2))
    
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    mean = np.mean(A)
    AM.append(mean)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    mean = np.mean(VA)
    VAM.append(mean)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    mean = np.mean(AA)
    AAM.append(mean)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    mean = np.mean(CM)
    CMM.append(mean)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    mean = np.mean(VCM)
    VCMM.append(mean)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    mean = np.mean(ACM)
    ACMM.append(mean)
    
    

    


# In[98]:


columns = ['Personne']


# In[99]:


index = [1,2,3,4,5,6]


# In[100]:


data = [1,2,3,4,5,6]


# In[101]:


df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[102]:


df['Vitesse quadratique moyenne'] = VM


# In[103]:


df['Accéleration quadratique moyenne']=AM


# In[104]:


df['vitesse angulaire moyenne']=VAM


# In[105]:


df['acceleration angulaire moyenne']=AAM


# In[106]:


df['centre de masse moyen']=CMM


# In[107]:


df['vitesse de centre de masse moyenne']=VCMM


# In[108]:


df['acceleration de centre de masse moyenne']=ACMM


# In[109]:


df['angle1']=AT1M


# In[110]:


df['angle2']=AT2M


# In[111]:


df


# In[47]:


sns.pairplot(df[['Personne','angle1','angle2']])


# In[48]:


'angle1','angle2','Vitesse quadratique moyenne','Accéleration quadratique moyenne','centre de masse moyen','vitesse de centre de masse moyenne','acceleration de centre de masse moyenne',


# In[112]:


df.drop(columns=['vitesse angulaire moyenne','acceleration angulaire moyenne'],inplace=True)


# In[113]:


df


# In[114]:


df.mean(axis=0)


# In[115]:


df.std(axis=0)


# In[116]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[117]:


dt.mean(axis=0)


# In[118]:


dt.std(axis=0)


# In[119]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[120]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt)


# In[121]:


model.labels_  #numéro de cluster affectés aux groupes


# In[122]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe 


# In[123]:


c=df.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# In[ ]:





# In[ ]:





# # Max 
# ## Mouvement : Extension du rachis
# Data : PRE & POST lombaligie \
# 8 angles

# In[80]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# In[81]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\PRE')


# In[82]:


VM=[]
AM=[]
VAM=[]
AAM=[]
CMM=[]
VCMM=[]
ACMM=[]
AT1M=[]
AT2M=[]
AT3M=[]
AT4M=[]
AT5M=[]
AT6M=[]
AT7M=[]
AT8M=[]

Files=['LIEURAIN\\LIEURAIN-003.xlsx','BUTSCHER\\BUTSCHER-002.xlsx','ADJADJ\\Extension_001.xlsx']
for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    
    AT1=angles.loc[:,angles.columns[10]] #Angles of T8
    AT2=angles.loc[:,angles.columns[11]] 
    AT3=angles.loc[:,angles.columns[12]] 
    AT4=angles.loc[:,angles.columns[13]] 
    AT5=angles.loc[:,angles.columns[15]] 
    AT6=angles.loc[:,angles.columns[16]] 
    AT7=angles.loc[:,angles.columns[17]] 
    AT8=angles.loc[:,angles.columns[18]] 
    
    AT1M.append(np.mean(AT1))
    AT2M.append(np.mean(AT2))
    AT3M.append(np.mean(AT3))
    AT4M.append(np.mean(AT4))
    AT5M.append(np.mean(AT5))
    AT6M.append(np.mean(AT6))
    AT7M.append(np.mean(AT7))
    AT8M.append(np.mean(AT8))
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    max = np.max(A)
    AM.append(max)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    max = np.max(VA)
    VAM.append(max)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    max = np.max(AA)
    AAM.append(max)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    max = np.max(CM)
    CMM.append(max)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    max = np.max(VCM)
    VCMM.append(max)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    max = np.max(ACM)
    ACMM.append(max)
    
    

    


# In[83]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\POST')


# In[84]:



Files=['LIEURAIN\\Emmanuel-002.xlsx','BUTSCHER\\BUTSCHER-YANN-002.xlsx','ADJADJA\\ADJADJA NICOLAS-004.xlsx']
for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    
    AT1=angles.loc[:,angles.columns[10]] #Angles of T8
    AT2=angles.loc[:,angles.columns[11]] 
    AT3=angles.loc[:,angles.columns[12]] 
    AT4=angles.loc[:,angles.columns[13]] 
    AT5=angles.loc[:,angles.columns[15]] 
    AT6=angles.loc[:,angles.columns[16]] 
    AT7=angles.loc[:,angles.columns[17]] 
    AT8=angles.loc[:,angles.columns[18]] 
    
    AT1M.append(np.mean(AT1))
    AT2M.append(np.mean(AT2))
    AT3M.append(np.mean(AT3))
    AT4M.append(np.mean(AT4))
    AT5M.append(np.mean(AT5))
    AT6M.append(np.mean(AT6))
    AT7M.append(np.mean(AT7))
    AT8M.append(np.mean(AT8))
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    max = np.max(V)
    VM.append(max)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    max = np.max(A)
    AM.append(max)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    max = np.max(VA)
    VAM.append(max)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    max = np.max(AA)
    AAM.append(max)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    max = np.max(CM)
    CMM.append(max)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    max = np.max(VCM)
    VCMM.append(max)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    max = np.max(ACM)
    ACMM.append(max)
    
    

    


# In[85]:


columns = ['Personne']


# In[86]:


index = [1,2,3,4,5,6]


# In[87]:


data = [1,2,3,4,5,6]


# In[88]:


df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[89]:


df['Vitesse quadratique moyenne'] = VM


# In[90]:


df['Accéleration quadratique moyenne']=AM


# In[91]:


df['vitesse angulaire moyenne']=VAM


# In[92]:


df['acceleration angulaire moyenne']=AAM


# In[93]:


df['centre de masse moyen']=CMM


# In[94]:


df['vitesse de centre de masse moyenne']=VCMM


# In[95]:


df['acceleration de centre de masse moyenne']=ACMM


# In[96]:


df['angle1']=AT1M


# In[97]:


df['angle2']=AT2M


# In[98]:


df['angle3']=AT3M


# In[99]:


df['angle4']=AT4M


# In[100]:


df['angle5']=AT5M


# In[101]:


df['angle6']=AT6M


# In[102]:


df['angle7']=AT7M


# In[103]:


df['angle8']=AT8M


# In[104]:


df


# In[105]:


sns.pairplot(df[['Personne','angle1','angle2','angle3','angle4','angle5','angle6','angle7','angle8']])


# In[106]:


df.mean(axis=0)


# In[107]:


df.std(axis=0)


# In[108]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[109]:


dt.mean(axis=0)


# In[110]:


dt.std(axis=0)


# In[111]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[112]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt)


# In[113]:


model.labels_  #numéro de cluster affectés aux groupes


# In[114]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe 


# In[116]:


c=dt.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Patinets lombaligues PRE et volontaires sains 
# ## Mouvement : Flexion du tronc
# Data : PRE lombaligie et volonatires sains de Chloé \
# 2 angles

# In[124]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\PRE')


# In[125]:


Files=['LIEURAIN\\LIEURAIN-001.xlsx','BUTSCHER\\BUTSCHER-001.xlsx','ADJADJ\\Flexion_001.xlsx','BENHAMMADI\\flexion_001.xlsx','GIMENO\\GIMEMO-001.xlsx']


# In[126]:


VM=[]
AM=[]
VAM=[]
AAM=[]
CMM=[]
VCMM=[]
ACMM=[]
AT1M=[]
AT2M=[]

for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    

    AT1=angles.loc[:,angles.columns[12]] #Angles of T8
    AT2=angles.loc[:,angles.columns[15]] 
    
    
    AT1M.append(np.max(AT1))
    AT2M.append(np.max(AT2))
    
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    mean = np.mean(A)
    AM.append(mean)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    mean = np.mean(VA)
    VAM.append(mean)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    mean = np.mean(AA)
    AAM.append(mean)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    mean = np.mean(CM)
    CMM.append(mean)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    mean = np.mean(VCM)
    VCMM.append(mean)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    mean = np.mean(ACM)
    ACMM.append(mean)
    
    


# In[ ]:





# In[127]:


os.chdir('C:\\Users\\Administrateur\\Desktop\\Spondylarthrite\\XSENS_SPONDYMOVE_ALL')


# In[128]:


Files=['002_BaEl\\002_BaEl-007.xlsx','010_AnKo\\010_AnKo-007.xlsx','022_DrGe\\022_DrGe-007.xlsx','005_LuWi\\005_LuWi-007.xlsx','019_GuAl\\019_GuAl-007.xlsx']


# In[129]:


for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")
    
    
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    
    VA1=vitesse_angulaire.loc[:,vitesse_angulaire.columns[13]] #Angular Velocity of T8
    VA2=vitesse_angulaire.loc[:,vitesse_angulaire.columns[14]]
    VA3=vitesse_angulaire.loc[:,vitesse_angulaire.columns[15]]
    
    
    AA1=acceleration_angulaire.loc[:,acceleration_angulaire.columns[13]] #Angular acceleration of T8
    AA2=acceleration_angulaire.loc[:,acceleration_angulaire.columns[14]]
    AA3=acceleration_angulaire.loc[:,acceleration_angulaire.columns[15]]
    
    
    CM1=centre_mass.loc[:,centre_mass.columns[1]] #Center of Mass 
    CM2=centre_mass.loc[:,centre_mass.columns[2]]
    CM3=centre_mass.loc[:,centre_mass.columns[3]]
    
    VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    

    AT1=angles.loc[:,angles.columns[12]] #Angles of T8
    AT2=angles.loc[:,angles.columns[15]] 
    
    
    AT1M.append(np.max(AT1))
    AT2M.append(np.max(AT2))
    
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    VCM=[]
    ACM=[]
    

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    mean = np.mean(A)
    AM.append(mean)

    
    for  k in range(len(VA1)) :
        VA.append (np.sqrt (VA1[k] * VA1[k] + VA2[k] *VA2[k] + VA3[k] *VA3[k]))
    mean = np.mean(VA)
    VAM.append(mean)
    
    
    for  k in range(len(AA1)) :
        AA.append (np.sqrt (AA1[k] * AA1[k] + AA2[k] *AA2[k] + AA3[k] *AA3[k]))
    mean = np.mean(AA)
    AAM.append(mean)
    
    
    for  k in range(len(CM1)) :
        CM.append (np.sqrt (CM1[k] * CM1[k] + CM2[k] *CM2[k] + CM3[k] *CM3[k]))
    mean = np.mean(CM)
    CMM.append(mean)
    
    for  k in range(len(VCM1)) :
        VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    mean = np.mean(VCM)
    VCMM.append(mean)
    
    for  k in range(len(ACM1)) :
        ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    mean = np.mean(ACM)
    ACMM.append(mean)
    
    


# In[146]:


columns = ['Personne']
index = [1,2,3,4,5,6,7,8,9,10]
data = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[147]:


df['Vitesse quadratique moyenne'] = VM
df['Accéleration quadratique moyenne']=AM
df['vitesse angulaire moyenne']=VAM
df['acceleration angulaire moyenne']=AAM
df['centre de masse moyen']=CMM
df['vitesse de centre de masse moyenne']=VCMM
df['acceleration de centre de masse moyenne']=ACMM
df['angle1']=AT1M
df['angle2']=AT2M


# In[148]:


df


# In[128]:


'Vitesse quadratique moyenne','Accéleration quadratique moyenne','acceleration de centre de masse moyenne','centre de masse moyen','vitesse de centre de masse moyenne','angle1','angle2'


# In[133]:


sns.pairplot(df[['Personne','angle1','angle2']])


# In[163]:


df.drop(columns=['Vitesse quadratique moyenne','Accéleration quadratique moyenne','vitesse angulaire moyenne','acceleration angulaire moyenne','acceleration de centre de masse moyenne','centre de masse moyen','vitesse de centre de masse moyenne'],inplace=True)


# In[164]:


df


# In[165]:


df.mean(axis=0)


# In[166]:


df.std(axis=0)


# In[167]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[168]:


dt.mean(axis=0)


# In[169]:


dt.std(axis=0)


# In[170]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[171]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt)


# In[172]:


model.labels_  #numéro de cluster affectés aux groupes


# In[173]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe 


# In[174]:


c=df.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import os
import FireFiles as ff
from PRG.SPARC_master.scripts.smoothness import sparc
import PATHS


def smoothness():
    paths = ff.get_all_files(PATHS.XSENS)
    excluded_paths = []
    exercises = ["001", "004", "007", "010"]

    for p in paths:
        if ".xlsx" in p and "Fluidite" not in p and "coupure" not in p and not "fluidite_repeat" in p:  # and "DoMa" in p:
            if os.path.basename(p).split(".")[0].split("-")[1] in exercises:
                excluded_paths.append(p)
    for p in excluded_paths:
        name = p.split("\\")[-1].split("_")[1].split(".")[0].split("-")[0]  # DoMa

        identity = os.path.basename(p).split(".")[0].split("-")[0]  # 003_DoMa
        exercise = os.path.basename(p).split(".")[0].split("-")[1]  # 001
        marker = name + "-" + exercise  # DoMa-001

        cut_name = identity + "-" + exercise + "_" + "Fluidite"

        cut_file = ff.get_all_files(PATHS.XSENS)
        coupure_paths = []
        for c in cut_file:
            if cut_name in c:
                coupure_paths.append(c)
        for coupure in coupure_paths:
            fluid_identity = os.path.basename(coupure).split("_")[2]
            number_fluidite = ""
            if "1" in fluid_identity:
                number_fluidite = "_go"
            elif "2" in fluid_identity:
                number_fluidite = "_back"
            cut_df = pd.read_excel(coupure)  # stock the cuts for each exercise
            cuts = cut_df.columns.to_list()[0].split(",")
            if cuts[-1] == '':
                cuts.pop()

            sheets_to_cut = ['Segment Velocity', ]

            cut_df = []  # [sheet1_cuts[[cut1], [cut2]...], sheet2_cuts[[cut1], [cut2]...]...] the whole exercise dataframe cut
            for sheet in sheets_to_cut:
                df_sheet = pd.read_excel(p, sheet_name=sheet)
                sheet_cuts = []
                current_cut = 0

                while current_cut < len(cuts) - 1:
                    low_cut = int(cuts[current_cut])
                    high_cut = int(cuts[current_cut + 1])
                    sheet_cuts.append(df_sheet.iloc[low_cut:high_cut])
                    current_cut += 1
                cut_df.append(sheet_cuts)

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                for c in range(len(cuts) - 1):
                    print(f"writing {identity}-{exercise}-repeat{c}.xlsx SHEET {sheet}")
                    new_path = rf'{PATHS.XSENS}\{identity}\fluidite_repeats'
                    ff.verify_dir(new_path)
                    writer = pd.ExcelWriter(new_path + rf"\{exercise}-fluidite_repeat{c}{number_fluidite}.xlsx",
                                            engine='xlsxwriter')

                    # Write each dataframe to a different worksheet.
                    n = 0
                    for sheet_cut in cut_df:
                        sheet_cut[c].to_excel(writer, sheet_name=sheets_to_cut[n])
                        n += 1

                    # Close the Pandas Excel writer and output the Excel file.
                    writer.save()


def analysis():
    identities_results = {"003_DoMa": ([], [], [],), "002_BaEl": ([], [], [],), "004_LoJu": ([], [], [],), "005_LuWi": ([], [], [],),
                          "006_ReCo": ([], [], [],), "007_AmVi": ([], [], [],), "008_GuEm": ([], [], [],), "009_ChCl": ([], [], [],),
                          "010_AnKo": ([], [], [],), "011_YaJu": ([], [], [],), "012_PaMa": ([], [], [],), "013_SaCl": ([], [], [],),
                          "014_MoLu": ([], [], [],), "015_LaJu": ([], [], [],), "016_LaCh": ([], [], [],), "017_GoMa": ([], [], [],),
                          "018_FaFr": ([], [], [],), "019_GuAl": ([], [], [],), "020_PlSi": ([], [], [],), "021_GuGi": ([], [], [],),
                          "022_DrGe": ([], [], [],), }
    paths = ff.get_all_files(PATHS.XSENS)
    excluded_paths = []
    doubled_exercise = ["007", "010"]
    exercises = ["001", "004", "007", "010"]
    markers = {"001": ["T8 z", "Right Hand z"], "004": ["T8 z", "Pelvis z"], "007": ["T8 z", "Right Hand z"],
               "010": ["Right Hand x", "T8 x"], }
    for p in paths:
        if ".xlsx" in p and "fluidite_repeat" in p:
            excluded_paths.append(p)

    for ex in exercises:
        all_results = []
        if ex not in doubled_exercise:
            for identity in identities_results:
                ex_repeats = []

                for excluded in excluded_paths:
                    if ex in excluded:
                        ex_repeats.append(excluded)

                for marker in markers[ex]:
                    all_repeats = []
                    fluidites = []
                    for repeat in ex_repeats:
                        if identity in repeat:
                            df = pd.read_excel(repeat)
                            all_repeats.append(df[marker])

                    arrays = [np.array(x) for x in all_repeats]
                    mean_signal = [np.mean(k) for k in zip(*arrays)]
                    std_signal = [np.std(k) for k in zip(*arrays)]

                    new_sal = 0
                    try:
                        new_sal = sparc(mean_signal, 60)[0]
                    except:
                        new_sal = 0
                    identities_results[identity][0].append(round(new_sal, 3))
                    all_results.append((identity, "none", marker, new_sal))

        else:
            for identity in identities_results:
                for b_g in ["_back", "_go"]:
                    ex_repeats = []
                    for excluded in excluded_paths:
                        if ex in excluded and b_g in excluded:
                            ex_repeats.append(excluded)

                    for marker in markers[ex]:
                        all_repeats = []
                        fluidites = []
                        for repeat in ex_repeats:
                            if identity in repeat:
                                df = pd.read_excel(repeat)
                                df = df.dropna(axis=0)
                                all_repeats.append(df[marker].to_list())
                        arrays = [np.array(x) for x in all_repeats]
                        mean_signal = [np.mean(k) for k in zip(*arrays)]
                        std_signal = [np.std(k) for k in zip(*arrays)]

                        new_sal = 0
                        try:
                            new_sal = sparc(mean_signal, 60)[0]
                        except:
                            new_sal = 0
                        identities_results[identity][0].append(round(new_sal, 3))
                        identities_results[identity][1].append(b_g[1:])
                        identities_results[identity][2].append(marker)
                        #all_results.append((identity, b_g[1:], marker, new_sal))

        print(identities_results)
        with open(rf"{PATHS.RES}\smoothness_{ex}.csv", "w+") as f:
            lines = []
            lines.insert(0, "identity,smoothness,orientation,marker\n")

            for id in identities_results:
                lines.append(f"{id},{identities_results[id][0]},{identities_results[id][1]},{identities_results[id][2]}\n")
            for line in lines:
                f.write(line)


# In[ ]:




