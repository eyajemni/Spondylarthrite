# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:01:52 2022

@author: dorey
"""

####Eya's Script for opening datafile
#Current working directory : where you should download the files
import os
os.getcwd()

#pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
import pandas as pd

###Reading the first sheet
data0=pd.read_excel("Pilot_006_JeEy_Ext.xlsx")
data0

###Reading the second sheet
data1=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name ="Markers")
data1

###Reading the thirdsheet
data2=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name =2)
data2

###Reading the fourth sheet
data3=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name =3)
data3


###Reading all sheets
data=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17])
data



data[17]

dataNone=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name = None)
dataNone


dataNone.keys() #return the names of sheets

dataName=pd.read_excel('Pilot_006_JeEy_Ext.xlsx', sheet_name = ["Segment Position","Segment Acceleration"])
dataName


#Other Method
excel= pd.ExcelFile('Pilot_006_JeEy_Ext.xlsx')
excel
excel.sheet_names

excel.parse('Ergonomic Joint Angles ZXY')



