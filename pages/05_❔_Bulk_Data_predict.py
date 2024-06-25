import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os
import datetime
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title='Predict Page',
    page_icon='🔍',
    layout='wide'
)


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

def load_XGBoost_pipeline():
    pipeline = joblib.load('./models/XGBoost_pipeline.joblib')
    return pipeline


st.cache_resource()
def load_Cat_Boost_pipeline():
    pipeline = joblib.load('./models/CatBoost_pipeline.joblib')
    return pipeline


st.cache_resource()
def GB_pipeline():
    pipeline = joblib.load('./models/GBC_pipeline.joblib')
    return pipeline


st.cache_resource()
def load_custom_imputer():
    return joblib.load('./models/custom_imputer.joblib')


st.cache_resource(show_spinner= 'Model Loading....')
def select_model():
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox('Select Model', options=['XGBoost', 'Gradient Boosting', 'Category Boosting'], key='selected_model')

    with col2:
        pass
    
    if st.session_state['selected_model'] == 'XGBoost':
        pipeline = load_XGBoost_pipeline()
        
    elif st.session_state['selected_model'] == 'Category Boosting':
        pipeline = load_Cat_Boost_pipeline()
    else:
        pipeline = GB_pipeline()
    
    encoder = joblib.load('./models/encoder.joblib')
    
    return pipeline, encoder

def make_predictions(pipeline, encoder, df):
    pred = pipeline.predict(df)
    pred_int = int(pred[0])
    prediction = encoder.inverse_transform([pred_int])[0]
    probability = pipeline.predict_proba(df)[0]
    return prediction, probability

def main():
    st.title("Prediction Application")

    # Initialize session state for prediction and probability
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None

    # Select model and encoder
    pipeline, encoder = select_model()

    # Display and download results
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    # Make predictions
        if st.button("Make Predictions"):
            prediction, probability = make_predictions(pipeline, encoder, df)
            if prediction is not None and probability is not None:
                st.session_state["prediction"] = prediction
                st.session_state["probability"] = probability
                df['Predictions'] = prediction if prediction is not None else 'N/A'

                if prediction == 'No':
                    df['Probability'] = st.session_state["probability"][0]
                else:
                    df['Probability'] = st.session_state["probability"][1]
                
                df['model_used'] = st.session_state['selected_model']

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
    else:
        st.warning("Please upload a CSV file.")

if st.session_state['authentication_status']:    
    authenticator.logout(location='sidebar')
    col1, col2 = st.columns(2)
    with col1:
        st.image('resources/churn image.png', width=200)
    with col2:
        st.header(':rainbow-background[Will customer Churn?]')

    main()


else:
    st.info('Login to gain access to the app')