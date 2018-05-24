# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:39:05 2018

@author: Michael
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import iqr
from sklearn import preprocessing

# Import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Merge test and train_no_target
target=train[['NumberOfCustomers','NumberOfSales']]
del train['NumberOfCustomers']
del train['NumberOfSales']
all_data=pd.concat([train,test])

# Extract events!
def ExtractEvents(df,col,sep):
    df[col]=df[col].astype('str').apply(lambda x: 'Nothing' if x=='nan' else x)
    events=list(df[col].unique())
    for event in events:
        if sep in event:
            event=event.split(sep)
            for i in event:
                events.append(i)
    events=list({x for x in events if sep not in x})
    for event in events:
        df[event]=df[col].astype('str').apply(lambda x: 1 if event in x else 0)
    del df[col]
    return df

all_data=ExtractEvents(all_data,'Events','-')
all_data[['Snow','Rain','Nothing']].head(10) #per esempio

# Fill max gust
all_data['Max_Gust_SpeedKm_h'] = all_data['Max_Gust_SpeedKm_h'].apply(lambda x: 0 if np.isnan(x) else x);

# Filtra IsOpen
all_data = all_data[all_data.IsOpen != 0]
all_data = all_data.drop(columns=['IsOpen'])

# Subdivide test and train
#train=all_data[:-test.shape[0]]
#train[['NumberOfCustomers','NumberOfSales']]=target
#test=all_data[train.shape[0]:]

new=all_data[['Max_Dew_PointC','Max_Gust_SpeedKm_h','Max_Humidity','Max_Sea_Level_PressurehPa',
             'Max_TemperatureC','Max_Wind_SpeedKm_h', 'Mean_Dew_PointC', 'Mean_Humidity',
             'Mean_Sea_Level_PressurehPa', 'Mean_TemperatureC','Mean_Wind_SpeedKm_h']]
# Missing values
new.describe()
# No missing lol


# Normalization and log(x_norm+1) - OCIO CI METTE UN PO'
# Ho normalizzato tutto perchè c'ho negativi!
minmax=preprocessing.MinMaxScaler()
for col in new.columns:
    normalized_x=minmax.fit_transform(new[col].values.reshape(-1,1))
    #normalized_x=preprocessing.scale(new[col].values.reshape(-1,1))
    normalized_x=np.log1p(normalized_x)
    new[col]=pd.DataFrame(normalized_x,index=range(1,new.shape[0]+1))
# Scatterplotting
new[['NumberOfCustomers','NumberOfSales']]=target
#new["NumberOfSales"]=np.log1p(new["NumberOfSales"])
for f in new:
    new.plot(kind="scatter", x=f, y="NumberOfSales")


# Idem ma con polinomi
new=all_data[['Max_Dew_PointC','Max_Gust_SpeedKm_h','Max_Humidity','Max_Sea_Level_PressurehPa',
             'Max_TemperatureC','Max_Wind_SpeedKm_h', 'Mean_Dew_PointC', 'Mean_Humidity',
             'Mean_Sea_Level_PressurehPa', 'Mean_TemperatureC','Mean_Wind_SpeedKm_h']]
n=5
for col in new.columns:
    new[col]=new[col]**n
#for col in new.columns:
#    new[col]=np.exp(new[col])
# Scatterplotting
new[['NumberOfCustomers','NumberOfSales']]=target
new["NumberOfSales"]=np.log1p(new["NumberOfSales"])
#normalized_y=minmax.fit_transform(new["NumberOfSales"].values.reshape(-1,1))
#new["NumberOfSales"]=pd.DataFrame(normalized_y,index=range(1,new.shape[0]+1))
#new["NumberOfSales"]=np.log10(new["NumberOfSales"])
#new["NumberOfSales"]=np.log(new["NumberOfSales"])*(new["NumberOfSales"]**2)
#new["NumberOfSales"]=np.log1p(new["NumberOfSales"])
for f in new:
    new.plot(kind="scatter", x=f, y="NumberOfSales")

new["NumberOfSales"].unique()
new["NumberOfSales"].value_counts()


# Considerazioni finali:
# - Meteo troppo rumoroso plottato con numero di vendite plottato con ogni
#   feature possibile immaginabile... l'unico modo per trovare una correlazione
#   tra i due sarebbe restringere il numero di vendite logaritmicamente, ma
#   neanche quello funzia...
#
# - Raga ma ci sono 6000 NumberOfSales=0 (1% del totale ripulito da IsOpen=0,
#   è normale?)