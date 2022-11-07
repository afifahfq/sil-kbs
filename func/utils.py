from flask import Flask, render_template, request
import pandas as pd, sklearn, matplotlib.pyplot as plt, numpy
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("Week09.csv")

df1 = df.drop(['Tanggal', 'Jam', 'Kota', 'Pembayaran'], axis=1)

df1['Total'] = pd.to_numeric(df1['Total'], errors='coerce')
df1 = df1.dropna()
df1 = df1.sort_values(by=['Item'])

dfk = df1.groupby(['Item'])
df2 = dfk.sum().reset_index()
qty = []
for i in df2['Item']:
    qty.append(len(dfk.get_group(i)))
df2['Qty'] = qty
df2_no_label = df2.drop(['Item'], axis=1)

clf = KMeans(n_clusters=3, random_state=0).fit(df2_no_label)

def get_products():    
    return df2['Item'].tolist()

def total(produk):
    return list(df2.loc[df2['Item'] == produk]['Total'])

def get_qty(produk):
    return list(df2.loc[df2['Item'] == produk]['Qty'])

def get_status(produk): 
    labels = clf.fit_predict(df2_no_label)
    cluster = labels[df2.loc[df2['Item'] == produk].index.values.astype(int)[0]]
    match cluster:
        case 0:
            return "Average"
        case 1:
            return "Low"
        case 2: 
            return "High"