import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import pickle
from typing import List
import json
st.set_page_config(layout="wide", page_title="Diabetes Prediction",
                   page_icon="💊", initial_sidebar_state="auto")


# Load the models and transformers once
MODEL_PATH = 'C:/Users/smartcopper/Desktop/PycharmProjects/projects/diabetes/forest_diabetes.pkl'
ONEHOT_PATH = 'C:/Users/smartcopper/Desktop/PycharmProjects/projects/diabetes/onehot_diabetes.pkl'
SCALER_PATH = 'C:/Users/smartcopper/Desktop/PycharmProjects/projects/diabetes/scaler_diabetes.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ONEHOT_PATH, 'rb') as f:
    onehot = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

cat_col = ['gender', 'smoking_history']
num_col = ['age', 'bmi', 'hemoglobin_level', 'blood_glucose_level']



st.title('Diabetes Prediction')
col1,col2=st.columns([3,1])
with col1:
     st.header('This is a simple app to predict diabetes')
with col2:
    st_lottie(json.load(open('C:/Users/smartcopper/Desktop/PycharmProjects/projects/diabetes/lottie file hospital report.json')),
                height=200,quality='medium',key="lottie")

with st.form(key='my_form'):
    col1,col2=st.columns([1,1],gap='large')
    with col1:
        gender=st.selectbox('Gender',('Male','Female'))
        age=st.slider('Age',0,100,step=1)
        hypertension=st.selectbox('Hypertension',(0,1))
        heart_disease=st.selectbox('Heart Disease',(0,1))
    with col2:
        smoking_history=st.selectbox('Smoking',('No Info','never','former','current','not current','ever'))
        bmi=st.number_input('BMI',min_value=0,max_value=100,value=10)
        hemoglobin_level=st.number_input('Hemoglobin Level',min_value=0,max_value=10)
        blood_glucose_level=st.slider('Blood Glucose Level',0,400,step=1)
    st.text(' ')
          

    def prediction(gender: int, age: int, hypertension: int, heart_disease: int, smoking_history: int, bmi: float, hemoglobin_level: float, blood_glucose_level: float) -> List[int]:
        # Initialize the DataFrame with the input values
        sample = pd.DataFrame(data=[[gender, age, hypertension, heart_disease, smoking_history, bmi, hemoglobin_level, blood_glucose_level]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'hemoglobin_level', 'blood_glucose_level'])

        # One-hot encode the categorical features
        sample_cat = pd.DataFrame(onehot.transform(sample[cat_col]))
        sample_cat.index = sample.index

        # Drop the categorical features and concatenate the one-hot encoded features
        sample = sample.drop(cat_col, axis=1)
        sample = pd.concat([sample, sample_cat], axis=1)

        # Scale the numerical features
        sample[num_col] = scaler.transform(sample[num_col])

        # Predict the class label and return it as a list
        predicted = model.predict(sample)
        return predicted[0]
    col1,col2,col3=st.columns([1,1,1])
    with col2:
        submit=st.form_submit_button('Predict',use_container_width=True)

col1,col2,col3=st.columns([1,1,1])
with col2:
    if submit:
        if prediction(gender,age,hypertension,heart_disease,smoking_history,bmi,hemoglobin_level,blood_glucose_level) == 1:
            st.subheader(':red[You have diabetes]')
        else:
            st.subheader('You do not have diabetes',anchor=None)
st.divider()
