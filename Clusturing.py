#!/usr/bin/env python
# coding: utf-8

# In[214]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# In[215]:


os.chdir("C:\\Users\\Administrateur\\Desktop\\Spondy")


# ## Calculer les vitesses quadratiques moyennes et accélerations quadratiques moyennes

# In[216]:


VM=[]
AM=[]
Files=['001_ChHo-001.xlsx','002_BaEl-001.xlsx','003_DoMa-001.xlsx','004_LoJu-001.xlsx','005_LuWi-001.xlsx','006_ReCo-001.xlsx','007_AmVi-001.xlsx','008_GuEm-001.xlsx','009_ChCl-001.xlsx','010_AnKo-001.xlsx','011_YaJu-001.xlsx','012_PaMa-001.xlsx','013_SaCl-001.xlsx','014_MoLu-001.xlsx','015_LaJu-001.xlsx','016_LaCh-001.xlsx','017_GoMa-001.xlsx','018_FaFr-001.xlsx','019_GuAl-001.xlsx','020_PlSi-001.xlsx','021_GuGi-001.xlsx','022_DrGe-001.xlsx']
for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    Frame=vitesse.loc[:,'Frame']

    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
    
    A1=acceleration.loc[:,acceleration.columns[13]] #Acceleration of T8
    A2=acceleration.loc[:,acceleration.columns[14]]
    A3=acceleration.loc[:,acceleration.columns[15]]

    V=[]
    A=[]

    for  k in range(len(V1)) :
        V.append (np.sqrt (V1[k] * V1[k] + V2[k] *V2[k] + V3[k] *V3[k]))
    mean = np.mean(V)
    VM.append(mean)
    
    for  k in range(len(A1)) :
        A.append (np.sqrt (A1[k] * A1[k] + A2[k] *A2[k] + A3[k] *A3[k]))
    mean = np.mean(A)
    AM.append(mean)


# In[217]:


AM


# # Construire la dataframe

# In[259]:


columns = ['Personne']


# In[260]:


index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]


# In[261]:


data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]


# In[262]:


df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[263]:


df['Vitesse quadratique moyenne'] = VM


# In[264]:


df['Accéleration quadratique moyenne']=AM


# In[265]:


df


# In[266]:


Age=[25,22,23,24,22,38,19,31,30,24,35,23,21,38,50,35,23,42,26,50,68,58]


# In[267]:


df['Age']=Age


# In[268]:


df


# In[269]:


Catégorie_age= []
for i in range (len (Age)) :
    if Age[i]<35 : Catégorie_age.append(1)
    elif (Age[i]>=35) & (Age[i]<55 ) : Catégorie_age.append(2)
    elif (Age[i]>=55) & (Age[i]<64 ) : Catégorie_age.append(3)
    elif Age[i]>=65 : Catégorie_age.append(4)


# In[270]:


Catégorie_age


# In[271]:


df['Catégorie_d_Age']=Catégorie_age


# In[272]:


df


# In[273]:


sns.pairplot(df[['Vitesse quadratique moyenne','Accéleration quadratique moyenne','Age','Catégorie_d_Age']])


# In[274]:


df.mean(axis=0)


# In[275]:


df.std(axis=0)


# In[277]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[278]:


dt.mean(axis=0)


# In[279]:


dt.std(axis=0)


# # Clusturing avec 2 classes

# In[320]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[321]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt)


# In[322]:


model.labels_  #numéro de cluster affectés aux groupes


# In[323]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe : 13 individus de la classe 0 et 9 de la classe 1


# In[324]:


c=df.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# # Clusturing avec 3 classes

# In[329]:


model2 = KMeans(n_clusters=3) #KMeans avec 3 groupes
model2.fit(dt)


# In[330]:


model2.labels_  #numéro de cluster affectés aux groupes


# In[331]:


np.unique(model2.labels_,return_counts=True) #effectifs par groupe 


# In[333]:


c=df.copy()
c['labels']=model2.labels_
sns.pairplot(c,hue="labels",palette={0:'b',1:'g',2:'r'})  #colorier les points dans le pairplot selon la classe


# # Clusturing avec 4 classes

# In[335]:


model3 = KMeans(n_clusters=4) #KMeans avec 4 groupes
model3.fit(dt)


# In[336]:


model3.labels_  #numéro de cluster affectés aux groupes


# In[337]:


np.unique(model3.labels_,return_counts=True) #effectifs par groupe 


# In[339]:


c=df.copy()
c['labels']=model3.labels_
sns.pairplot(c,hue="labels",palette={0:'b',1:'g',2:'r',3:'y'})  #colorier les points dans le pairplot selon la classe


# ## Classification (avec y= catégorie d'age)

# In[377]:


y=df.Catégorie_d_Age
X=df.drop(['Catégorie_d_Age'],axis=1)


# In[378]:


X_train , X_test, y_train, y_test = train_test_split (X,y)


# In[379]:


model4 = LogisticRegression (tol =0.1)


# In[380]:


model4.fit(X_train,y_train)


# In[381]:


y_predict=model4.predict(X_test)
y_predict


# In[382]:


y_test = np.array(y_test)
y_test


# In[383]:


from sklearn.model_selection import cross_val_score


# In[384]:


a= cross_val_score(model4,X_train,y_train, scoring="accuracy")


# In[385]:


a.mean() #accuracy of the model 


# In[ ]:




