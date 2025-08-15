ML Fraud Detector
==================
A machine learning application for detecting potential fraudulent transactions.

Features:
- scikit-learn pipeline (data preprocessing + logistic regression)
- Model training and evaluation with classification metrics
- Streamlit web app for interactive probability checks
- Synthetic dataset generation for testing
- Model persistence with joblib

Installation & Run:
-------------------
pip install -r requirements.txt
python train.py
streamlit run app.py

Usage:
- Adjust transaction parameters in UI sliders
- View fraud probability and classification output

Purpose:
--------
Demonstrates ML pipeline building, model deployment, and UI integration for real-time predictions.
