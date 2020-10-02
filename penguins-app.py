import streamlit as st
import pandas as pd
import pickle

st.write("""
# Simple Breast Cancer Detection App

This app predicts the **Breast Cancer** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Mean_Perimeter = st.sidebar.slider('Mean Perimeter', 35.0,180.0,122.8)
    Mean_Area = st.sidebar.slider('Mean Area', 120.0, 1900.0,1001.0)
    Area_Error = st.sidebar.slider('Area Error', 7.0, 200.0,153.4)
    Worst_Perimeter = st.sidebar.slider('Worst Perimeter', 40.0, 220.0,184.6)
    Worst_Area = st.sidebar.slider('Worst Area', 170.0,3500.0,2019.0)
    data = {'Mean Perimeter': Mean_Perimeter,
            'Mean Area': Mean_Area,
            'Area Error': Area_Error,
            'Worst Perimeter': Worst_Perimeter,
            'Worst Area':Worst_Area}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

clf = pickle.load(open('penguins_clf.pkl', 'rb'))

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)
mapping={0:'malignant',1:'benign'}
st.subheader('Prediction')
st.write(mapping[prediction[0]])



st.subheader('Prediction Probability')
st.write(prediction_proba)
