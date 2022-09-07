#!/usr/bin/env python
# coding: utf-8

# # Mouvement : Ramasser objet 
# ## Data : Patients lombaligues et volontaires sains 

# In[321]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# In[322]:


os.chdir('C:\\Users\\Eya\\Desktop\\Xsens\\LOMBALGIE\\sujets_lombalgiques_mars_2022\\PRE')
#La méthode Python os.chdir() remplace le répertoire de travail actuel par le chemin donné.


# In[323]:


#La liste files contient les noms des fichiers cotenant les données des sujets lombalgiques

Files=['LIEURAIN\\LIEURAIN-006.xlsx','BUTSCHER\\BUTSCHER-005.xlsx','ADJADJ\\ramasser_un_object_001.xlsx','BENHAMMADI\\RamaserObjet.xlsx','GIMENO\\GIMEMO-005.xlsx']


# In[324]:


#Ce bloc concerne les données des personnes lombalgiques


VM=[]
AM=[]
VAM=[]
AAM=[]
CMM=[]
VCMM=[]
ACMM=[]
AT1M=[]
AT2M=[]


#Toutes les listes initialisés à [], prenderont les valeurs quadratiques moyennes des paramétres suivants : 
#vitesse, accélération, vitesse angulaire, accélération angulaire, centre de masse, vitesse de centre de masse, 
#accélération de centre de masse, angle 1 et angle2.

#Les angles 1 et 2 correspondent au ROM, qui est l'Amplitude du mouvement (de l’anglais Range Of Motion).
#Elles renseignent sur la capacité d’une articulation à effectuer l’ensemble de ses mouvements.

#Pour le mouvement "Extension du rachis", les deux angles sont : "Angle de flexion maximal entre la verticale et le pelvis",
#et "Angle d’extension maximal entre la verticale et le thorax (T8)".


for i in range (len(Files)) :
    File=Files[i]
    vitesse=pd.read_excel(File, sheet_name ="Segment Velocity")
    acceleration=pd.read_excel(File, sheet_name ="Segment Acceleration")
    vitesse_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Velocity")
    acceleration_angulaire=pd.read_excel(File, sheet_name ="Segment Angular Acceleration")
    centre_mass=pd.read_excel(File, sheet_name ="Center of Mass")
    angles=pd.read_excel(File, sheet_name ="Ergonomic Joint Angles ZXY")

    
# A chaque fichier de la liste Files, les sheets à utiliser sont extractés dans des variables.
    
    
    vitesse.drop( vitesse[ (vitesse['Frame'] > 3000) ].index, inplace=True)
    acceleration.drop( acceleration[ (acceleration['Frame'] > 3000) ].index, inplace=True)
    vitesse_angulaire.drop( vitesse_angulaire[ (vitesse_angulaire['Frame'] > 3000) ].index, inplace=True)
    acceleration_angulaire.drop( acceleration_angulaire[ (acceleration_angulaire['Frame'] > 3000) ].index, inplace=True)
    centre_mass.drop( centre_mass[ (centre_mass['Frame'] > 3000) ].index, inplace=True)
    angles.drop( angles[ (angles['Frame'] > 3000) ].index, inplace=True)

    
# Nous avons fixés la valeur maximale des frames dans chaque fichier à 3000, pour que tous les fichiers soient de meme taille.   
    
    
    Frame=vitesse.loc[:,'Frame']

    
    V1=vitesse.loc[:,vitesse.columns[13]] #Velocity of T8
    V2=vitesse.loc[:,vitesse.columns[14]]
    V3=vitesse.loc[:,vitesse.columns[15]]
 
# On extracte dans V1, les vitesses correspondantes à l'axe des x, dans V2 celles correspondantes à l'axe des y et dans V3
#celles correspondantes à l'axe des z.

# De meme pour les autres
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
    
# VM contient les vitesses quadratiques moyennes
# De meme pour les autres    
    
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
    
    


# In[325]:


os.chdir('C:\\Users\\Eya\\Desktop\\Xsens\\LOMBALGIE\\SUJETS_SAINS')


# In[326]:


Files=['thierry\\thierry-006.xlsx' , 'Unai G\\Unai-007.xlsx' , 'Isabel Tavares\\RamasserObjet_001.xlsx','estelle ARGOTTI\\estelle-008.xlsx' , 'Barbara Rider\\TEST-005.xlsx']


# In[327]:


#Ce bloc concerne les données des personnes sains 

# Le meme démarche a été fait comme le bloc précédent.

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
    
    #VCM1=centre_mass.loc[:,centre_mass.columns[4]] #Velocity of Center of Mass 
    #VCM2=centre_mass.loc[:,centre_mass.columns[5]]
    #VCM3=centre_mass.loc[:,centre_mass.columns[6]]
    
    #ACM1=centre_mass.loc[:,centre_mass.columns[7]] #Acceleration of Center of Mass 
    #ACM2=centre_mass.loc[:,centre_mass.columns[8]]
    #ACM3=centre_mass.loc[:,centre_mass.columns[9]]
    
    

    AT1=angles.loc[:,angles.columns[12]] #Angles of T8
    AT2=angles.loc[:,angles.columns[15]] 
    
    
    AT1M.append(np.max(AT1))
    AT2M.append(np.max(AT2))
    
    
    V=[]
    A=[]
    VA=[]
    AA=[]
    CM=[]
    #VCM=[]
    #ACM=[]
    

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
    
    #for  k in range(len(VCM1)) :
    #    VCM.append (np.sqrt (VCM1[k] * VCM1[k] + VCM2[k] *VCM2[k] + VCM3[k] *VCM3[k]))
    #mean = np.mean(VCM)
    #VCMM.append(mean)
    #
    #for  k in range(len(ACM1)) :
    #    ACM.append (np.sqrt (ACM1[k] * ACM1[k] + ACM2[k] *ACM2[k] + ACM3[k] *ACM3[k]))
    #mean = np.mean(ACM)
    #ACMM.append(mean)
    
    


# ### Construction de la base de données

# In[328]:


# Création de la colonne Personne
columns = ['Personne']
index = [1,2,3,4,5,6,7,8,9,10]
data = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(data=data,index=index,columns=columns)
df


# In[329]:


# Création des autres colonnes de la base

df['Vitesse quadratique moyenne'] = VM
df['Accéleration quadratique moyenne']=AM
df['vitesse angulaire moyenne']=VAM
df['acceleration angulaire moyenne']=AAM
df['centre de masse moyen']=CMM
#df['vitesse de centre de masse moyenne']=VCMM
#f['acceleration de centre de masse moyenne']=ACMM
df['angle1']=AT1M
df['angle2']=AT2M


# In[330]:


df


# In[331]:


# Tracer la relation entre les deux angles

sns.pairplot(df[['Personne','angle1','angle2']])


# In[332]:


#df.drop(columns=['vitesse angulaire moyenne','acceleration angulaire moyenne'],inplace=True)
df


# ## Standardisation des données

# In[333]:


df.mean(axis=0) #moyenne par colonne


# In[334]:


df.std(axis=0) #écartype par colonne


# In[335]:


dt = (df - df.mean(axis=0)) / (df.std(axis=0))  # pour la standardisation de la dataframe
dt


# In[336]:


dt.mean(axis=0)


# In[337]:


dt.std(axis=0) #Les écartypes des colonnes sont bien égales à 1


# # Apprentissage non supervisé
# ## Application du modéle KMeans

# In[338]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[339]:


model = KMeans(n_clusters=2) #KMeans avec 2 groupes
model.fit(dt) # fit du  Training set


# In[340]:


model.labels_  #affiche le numéro de cluster affectés aux groupes


# In[341]:


np.unique(model.labels_,return_counts=True) #effectifs par groupe 


# In[342]:


c=df.copy()
c['labels']=model.labels_
sns.pairplot(c,hue="labels")  #colorier les points dans le pairplot selon la classe


# # Apprentissage supervisé

# In[343]:


classe=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Ajout de la colonne classe, qui sera le target dans notre apprentissage
# 0 : indique malade 
# 1 : indique sain


# In[344]:


df['classe']=classe
df


# In[345]:


# Particition de la base en target y (classe) et données x 

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


# In[346]:


y_train=model.fit(x_train) # fit du  Training set 
y_pred= model.predict(x_test) # tester le modéle en predire les targets test


# In[347]:


accurancy=np.sum(y_pred==y_test)/len(y_test) # Performance du modéle


# In[348]:


print(accurancy)


# In[349]:


y_pred


# In[ ]:





# ## Modéle SVM

# In[350]:


df


# In[351]:


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/2, random_state=0)


# In[352]:


# Fit du  Training set
from sklearn.svm import SVC
model2 = SVC(kernel = 'linear', random_state = 0)
model2.fit(X_train, y_train)

#Prediction sur le Test set
y_pred = model2.predict(X_test)


# In[353]:


accurancy=np.sum(y_pred==y_test)/len(y_test)
print(accurancy)


# In[ ]:





# ## Modéle Regréssion Logistique

# In[354]:


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#spliting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y)


# In[355]:


model3 = LogisticRegression (tol =0.1)


# In[356]:


model3.fit(X_train,y_train)


# In[357]:


y_predict=model3.predict(X_test)
y_predict


# In[358]:


y_test = np.array(y_test)
y_test


# In[359]:


accurancy=np.sum(y_predict==y_test)/len(y_test)
accurancy


# In[ ]:





# In[ ]:




