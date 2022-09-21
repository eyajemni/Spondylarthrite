#!/usr/bin/env python
# coding: utf-8

# # Mouvement : Flexion du tronc
# ## Data : Patients lombaligues et volontaires sains 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# In[2]:


os.chdir('C:\\Users\\Eya\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\PRE')


# In[3]:


Files=['LIEURAIN\\LIEURAIN-001.xlsx','BUTSCHER\\BUTSCHER-001.xlsx','ADJADJ\\Flexion_001.xlsx','BENHAMMADI\\flexion_001.xlsx','GIMENO\\GIMEMO-001.xlsx']


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
    
    


# In[5]:


os.chdir('C:\\Users\\Eya\\Desktop\\Xsens\\LOMBALGIE\\SUJETS_SAINS')


# In[6]:


Files=['thierry\\thierry-001.xlsx' , 'Unai G\\Unai-001.xlsx' , 'Isabel Tavares\\FlexionDos_001.xlsx','estelle ARGOTTI\\estelle-001.xlsx' , 'Barbara Rider\\TEST-001.xlsx']


# In[7]:


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
    
    


# In[8]:


columns = ['Personne']
index = [1,2,3,4,5,6,7,8,9,10]
data = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[9]:


df['Vitesse quadratique moyenne'] = VM
df['Accéleration quadratique moyenne']=AM
df['vitesse angulaire moyenne']=VAM
df['acceleration angulaire moyenne']=AAM
df['centre de masse moyen']=CMM
df['vitesse de centre de masse moyenne']=VCMM
df['acceleration de centre de masse moyenne']=ACMM
df['angle1']=AT1M
df['angle2']=AT2M


# In[70]:


df


# In[71]:


'Vitesse quadratique moyenne','Accéleration quadratique moyenne','acceleration de centre de masse moyenne','centre de masse moyen','vitesse de centre de masse moyenne','angle1','angle2'


# In[72]:


#df.drop(columns=['vitesse de centre de masse moyenne','acceleration de centre de masse moyenne'],inplace=True)


# In[73]:


sns.pairplot(df[['Personne','angle1','angle2']])


# In[74]:


df.mean(axis=0)


# In[75]:


df.std(axis=0)


# In[76]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[77]:


dt.mean(axis=0)


# In[78]:


dt.std(axis=0)


# # Apprentissage non supervisé
# ## Application du modéle KMeans

# In[79]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[80]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt)


# In[81]:


model.labels_  #numéro de cluster affectés aux groupes


# In[82]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe 


# In[83]:


c=df.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# # Apprentissage supervisé

# In[84]:


classe=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# In[85]:


df['classe']=classe
df


# In[86]:


# Particition de la base en target y (classe) et données x 

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


# In[87]:


y_train=model.fit(x_train) # fit du  Training set 
y_pred= model.predict(x_test) # tester le modéle en predire les targets test


# In[88]:


accurancy=np.sum(y_pred==y_test)/len(y_test) # Performance du modéle
print(accurancy)


# ## Modéle SVM

# In[89]:


df


# In[90]:


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/2, random_state=0)


# In[91]:


# Fit du  Training set
from sklearn.svm import SVC
model2 = SVC(kernel = 'linear', random_state = 0)
model2.fit(X_train, y_train)

#Prediction sur le Test set
y_pred = model2.predict(X_test)


# In[92]:


from sklearn.metrics import accuracy_score


# In[93]:


print(accuracy_score(y_test,y_pred)) # Performance du modéle


# ## Modéle Regréssion Logistique

# In[94]:


df['classe'] = [0,0,0,0,0,1,1,1,1,1]
df


# In[95]:


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y)


# In[96]:


model3 = LogisticRegression (tol =0.1)


# In[97]:


model3.fit(X_train,y_train)


# In[98]:


y_predict=model3.predict(X_test)
y_predict


# In[99]:


y_test = np.array(y_test)
y_test


# In[100]:


print(accuracy_score(y_test,y_predict)) # Performance du modéle

