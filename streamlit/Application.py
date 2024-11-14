import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

## ---- Data ---- ##

path = "/Users/jdgomez/UVG/data-science/proj2-data-science"

data = pd.read_csv(f'{path}/result/data_cleaned.csv')

st.title('Data Analysis')


## ---- Information ---- ##

category = {'discourse_type': False, 'discourse_effectiveness': False, 'text_length': True, 'word_count': True}
labels = ["Effective", "Adequate", "Ineffective"]
label_mapping_effectiveness = {
    "Effective": 0,
    "Adequate": 1,
    "Ineffective": 2
}
discourseTypes = ["Claim", "Concluding Statement", "Counterclaim", "Evidence", "Lead", "Position", "Rebuttal"]
label_mapping_types = {
    "Claim": 0,
    "Concluding Statement": 1,
    "Counterclaim": 2,
    "Evidence": 3,
    "Lead": 4,
    "Position": 5,
    "Rebuttal": 6
}


## ---- Sidebar ---- ##

st.sidebar.title('Variables')

# Categorical variable

categorical_variable = st.sidebar.selectbox("Variables", data.columns[3:5], index=0, placeholder="Select contact method...")

selected_values = []
selected_filter = []

if categorical_variable == 'discourse_effectiveness':
    options = labels
    selected_values = st.sidebar.multiselect(categorical_variable, options=options, default=options)
    selected_filter = [label_mapping_effectiveness[label] for label in selected_values]
    print("selected_values", selected_values)
    print("variable_filter", selected_filter)
else:
    options = discourseTypes
    selected_values = st.sidebar.multiselect(categorical_variable, options=options, default=options)
    selected_filter = [label_mapping_types[label] for label in selected_values]
    

# Numerical variable

numerical_variable = st.sidebar.selectbox("Variables", data.columns[5:7], index=0, placeholder="Select contact method...")

values = data[numerical_variable]
values_slider = st.sidebar.slider(numerical_variable, min_value=values.min(), max_value=values.max(), value=(values.min(), values.max()))


## ---- Data Filtering ---- ##

# Filter the data based on the selected variables
data_filtered = data[[categorical_variable, numerical_variable]]

# Filter the data based on the selected values of categorical
data_filtered = data_filtered.query(f"{categorical_variable} in @selected_filter")

# Filter the data based on the selected values of numerical
data_filtered = data_filtered[data_filtered[numerical_variable].between(values_slider[0], values_slider[1])]


## ---- Plots ---- ##

category_counts = data_filtered[categorical_variable].value_counts()

# Create the pie chart
fig_1 = px.pie(values=category_counts, names=selected_values, title='Pie Chart', color=category_counts.index, color_discrete_map={'Effective': 'green', 'Adequate': 'blue', 'Ineffective': 'red'})
st.plotly_chart(fig_1)


data_filtered = data_filtered.groupby(categorical_variable).mean().reset_index()

# Create the bar chart
fig_2 = px.bar(data_filtered, x=categorical_variable, y=numerical_variable, title='Bars', labels={numerical_variable: f'{numerical_variable} Promedio'})
st.plotly_chart(fig_2)
