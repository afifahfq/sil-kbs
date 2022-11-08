from flask import Flask, render_template, request, jsonify
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.cluster import KMeans
import jsonpickle
import func

app = Flask(__name__)
# ==================Coba coba di Jinjaaaaah==================
# rendering the template


@app.route("/")
def index():
    option = func.get_products()
    return render_template("index.html", option=option)


@app.route("/process")
def process():
    data = request.values
    total = func.total(data['item'])
    qty = func.get_qty(data['item'])
    status = func.get_status(data['item'])
    result = {'status': status,
              'qty': qty[0],
              'total': total[0]}
    return jsonify(result)
# ====================End of Jinjaaaaah=====================


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


# ============================FOR REACT====================================
# API endpoint
@app.route("/products")
def get_products():
    return df2['Item'].tolist()

# get endpoint


@app.route("/clusters")
def get_all_clusters():
    return {"clusters": clf.cluster_centers_.tolist()}


@app.route("/total")
def total():
    args = request.args
    produk = args.get('produk')
    return {"qty": list(df2.loc[df2['Item'] == produk]['Total'])}


@app.route("/qty")
def get_qty():
    args = request.args
    produk = args.get('produk')
    return {"qty": list(df2.loc[df2['Item'] == produk]['Qty'])}


@app.route("/cluster")
def get_cluster():
    args = request.args
    produk = args.get('produk')
    labels = clf.fit_predict(df2_no_label)
    print(clf.cluster_centers_[
          labels[df2.loc[df2['Item'] == produk].index.values.astype(int)[0]]])
    return {"cluster": list(clf.cluster_centers_[labels[df2.loc[df2['Item'] == produk].index.values.astype(int)[0]]])}


@app.route("/status")
def get_status():
    args = request.args
    produk = args.get('produk')
    labels = clf.fit_predict(df2_no_label)
    cluster = labels[df2.loc[df2['Item'] == produk].index.values.astype(int)[
        0]]
    match cluster:
        case 0:
            return {"status": "Average"}
        case 1:
            return {"status": "Low"}
        case 2:
            return {"status": "High"}


if __name__ == "__main__":
    app.run(debug=True)
