from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app= Flask(__name__)

model=load_model("Cancer_model.h5")

scalar=StandardScaler()
data=pd.read_csv('data/Breast_cancer.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
sample_feature=data['diagnosis']
scalar.fit(sample_feature)