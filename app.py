import streamlit as st, pandas as pd
from joblib import load
st.set_page_config(page_title='Fraud Detector', layout='wide')
st.title('🕵️ Fraud Detector Demo')
model = load('model.joblib')
amount = st.slider('Amount', 0.0, 500.0, 80.0, 1.0)
hour = st.slider('Hour of day', 0, 23, 12, 1)
merchant_risk = st.slider('Merchant risk', 0.0, 1.0, 0.4, 0.01)
distance_km = st.slider('Distance km', 0.0, 200.0, 10.0, 0.5)
df = pd.DataFrame([{"amount":amount,"hour":hour,"merchant_risk":merchant_risk,"distance_km":distance_km}])
proba = model.predict_proba(df)[0][1]
st.metric("Fraud probability", f"{proba:.2%}")
st.caption("Model: LogisticRegression on synthetic features; demo only.")
