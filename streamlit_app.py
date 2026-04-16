import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Alura Bank - Churn Predictor", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stVerticalBlock"] > div:has(div.stSlider) {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background-color: #004a99;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
