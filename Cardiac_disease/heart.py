import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

st.write("""
# Heart Disease Analysis

This app predicts Whether a person has Heart Disease or not
""")

Disease_raw=pd.read_csv('heart_disease_data.csv')
Disease_raw.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'old_peak', 'st_slope','target']
x=Disease_raw.drop('target',axis=1)
y=Disease_raw['target']

st.sidebar.header('User Input Parameter')

def user_input():
  age=st.sidebar.slider('Age',10,80,60)
  sex=st.sidebar.slider('Gender',0,1,1)
  chest_pain_type=st.sidebar.slider('Chest Pain Type',1,4,2)
  resting_blood_pressure=st.sidebar.slider('BP',80,180,170)
  cholesterol=st.sidebar.slider('Cholestrol',0,500,250)
  fasting_blood_sugar =st.sidebar.slider('Blood Sugar',0,1,0)
  rest_ecg =st.sidebar.slider('ECG',0,2,1)
  max_heart_rate_achieved =st.sidebar.slider('Heart Rate',80,180,100)
  exercise_induced_angina =st.sidebar.slider('Induced Angina',0,1)
  old_peak=st.sidebar.slider('Old Peak',0,3,1)
  st_slope =st.sidebar.slider('Slope',1,2,1)
  data={'age':age, 'sex':sex, 'chest_pain_type':chest_pain_type, 'resting_blood_pressure':resting_blood_pressure,
        'cholesterol':cholesterol, 'fasting_blood_sugar':fasting_blood_sugar, 'rest_ecg':rest_ecg, 'max_heart_rate_achieved':max_heart_rate_achieved,
        'exercise_induced_angina': exercise_induced_angina, 'old_peak': old_peak,'st_slope':st_slope
      }
  features=pd.DataFrame(data,index=[0])
  return features
df=user_input()

st.subheader('User Input Features')
st.write('Change the slider to modify the value and obtain outputs')
st.write(df)
st.write('---')

model=RandomForestClassifier()
model.fit(x, y)
prediction= model.predict(df)

probability = model.predict_proba(df)
st.subheader("Prediction")
st.write(prediction)

st.subheader('Probability')
st.write(probability)
