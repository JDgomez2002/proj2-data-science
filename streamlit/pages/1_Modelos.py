import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

## ---- Data ---- ##

data = pd.read_csv('result/data_cleaned.csv')

st.title('Data Analysis')
