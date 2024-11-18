import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors

## ---- Data ---- ##

path = "/Users/jdgomez/UVG/data-science/proj2-data-science"

data = pd.read_csv(f'result/data_cleaned.csv')

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
    selected_filter = selected_values
else:
    options = discourseTypes
    selected_values = st.sidebar.multiselect(categorical_variable, options=options, default=options)
    selected_filter = selected_values

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
pieTitle = ''
if categorical_variable == 'discourse_effectiveness':
    pieTitle = 'Discourse Effectiveness distribution'
    color_map = {
        'Effective': '#2D9494',  # Teal
        'Adequate': '#CDECAC',   # Light green
        'Ineffective': '#CC5A49' # Salmon
    }
else:
    pieTitle = 'Discourse type distribution'
    color_map = {
        'Claim': '#000000',           # Black
        'Concluding Statement': '#588157',    # White
        'Counterclaim': '#E5E5E5',    # Light gray
        'Evidence': '#CDECAC',        # Light green
        'Lead': '#2D9494',           # Teal
        'Position': '#CC5A49',        # Salmon
        'Rebuttal': '#344E41'         # White
    }

fig_1 = px.pie(
    values=category_counts,
    names=selected_values,
    title=pieTitle,
    color=category_counts.index,
    color_discrete_map=color_map
)

st.plotly_chart(fig_1)

# data_filtered2 = data_filtered.groupby(categorical_variable)
data_filtered = data_filtered.groupby(categorical_variable).mean().reset_index()

# Create the bar chart

barTitle = ''
if categorical_variable == 'discourse_effectiveness':
    barTitle = 'Mean text length per discourse effectiveness'
    color_sequence = ['#2D9494', '#CDECAC', '#CC5A49']  # Teal, Light green, Salmon
else:
    barTitle = 'Mean text length per discourse type'
    color_sequence = [ '#A3B18A', '#CDECAC', '#2D9494', '#CC5A49', '#344E41']  # Full palette

fig_2 = px.bar(
    data_filtered,
    x=categorical_variable,
    y=numerical_variable,
    title=barTitle,
    labels={numerical_variable: f'Mean {numerical_variable}'},
    color=categorical_variable,
    color_discrete_sequence=color_sequence
)

# Update layout for better visibility
fig_2.update_layout(
    plot_bgcolor='rgba(0,0,0,0.05)',
    showlegend=False
)

# Add gridlines
fig_2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
fig_2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

st.plotly_chart(fig_2)

# Scatter plot

scatTitle = ''
if categorical_variable == 'discourse_effectiveness':
    scatTitle = 'Text length count per discourse effectiveness'
else:
    scatTitle = 'Text length count per discourse type'

fig_3 = px.scatter(data_filtered, x=numerical_variable, y=categorical_variable, title=scatTitle, color=categorical_variable)
st.plotly_chart(fig_3)

fig_4 = px.scatter(data, x='text_length', y='word_count',
                 color='discourse_effectiveness', 
                 symbol='discourse_type',  # Agregar símbolos para discourse_type
                 title='Text length & word count relation by discourse effectiveness & discourse type',
                 labels={'text_length': 'Text Length', 'word_count': 'Word Count'},
                 hover_data=['discourse_type'])  # Mostrar información adicional al pasar el cursor

fig_4.update_layout(legend_title_text='Discourse Effectiveness y Discourse Type')

st.plotly_chart(fig_4)

## ---- Word Analysis ---- ##

# Analyzing the text column (assuming 'text' is the column name)
if 'discourse_text' in data.columns:
    all_words = ' '.join(data['discourse_text'].dropna()).split()
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(10)
    st.subheader("Top 10 Most Common Words")
    color_sequence = ['#CDECAC', '#2D9494', '#CC5A49', '#344E41']  # Full palette

    # Crear el gráfico de barras para las 10 palabras más comunes
    common_words_df = pd.DataFrame(most_common_words, columns=["Word", "Count"])
    fig_4 = px.bar(common_words_df, x='Word', y='Count', title='Top 10 Most Common Words', color_discrete_sequence=color_sequence)
    st.plotly_chart(fig_4)

    # Word Cloud with custom colormap
    st.subheader("Word Cloud")
    color_map = {
        0: "#000000",  # Black
        0.2: "#A3B18A",  # White
        0.4: "#588157",  # Light gray
        0.6: "#3A5A40",  # Light green
        0.8: "#2D9494",  # Teal
        1.0: "#CC5A49"  # Salmon
    }

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom",
            list(color_map.values())
        )
    ).generate(' '.join(all_words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
else:
    st.warning("No 'discourse_text' column found in the dataset to analyze words.")

## ---- Exploratory analysis graphs ---- ##

# correlation matrix for the variables correlation

# Correlation matrix
st.subheader("Variables correlation matrix")

numeric_cols = ['text_length', 'word_count']
categorical_cols = ['discourse_type', 'discourse_effectiveness']

# Create dummy variables for categorical columns
df_encoded = pd.get_dummies(data[categorical_cols + numeric_cols], columns=categorical_cols)

# Calculate correlation matrix
corr_matrix = df_encoded.corr()

# Create heatmap using custom color palette
fig_corr = px.imshow(
    corr_matrix,
    color_continuous_scale=['#000000', '#FFFFFF', '#E5E5E5', '#CDECAC', '#2D9494', '#CC5A49'],
    labels=dict(color="Correlation"),
    title=" "
)

# Update layout for better readability
fig_corr.update_traces(text=corr_matrix.round(2), texttemplate='%{text}')
fig_corr.update_layout(
    width=1000,
    height=600,
    title_x=0.5,
    title_y=0.95
)

# Add the plot to Streamlit
st.plotly_chart(fig_corr)

# top 10 frequent words per discourse type (7 discourse types in total)

## ---- Word frequency analysis by discourse type ---- ##

st.subheader("Most common words per Discourse Type")

# Color palette from the image
colors = ['#000000', '#FFFFFF', '#E5E5E5', '#CDECAC', '#2D9494', '#CC5A49']

# Create word frequency charts for each discourse type
for i, dtype in enumerate(discourseTypes):
    # Filter data for current discourse type
    mask = data['discourse_type'] == dtype
    text = ' '.join(data[mask]['discourse_text'].dropna())

    # Count word frequencies
    words = text.split()
    word_freq = Counter(words)
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    # Create DataFrame
    df_words = pd.DataFrame({
        'Word': list(top_words.keys()),
        'Frequency': list(top_words.values())
    })

    # Create bar chart
    fig = px.bar(
        df_words,
        x='Word',
        y='Frequency',
        title=f'Top 10 words for {dtype}',
        color_discrete_sequence=[colors[4]]  # Using teal color from palette
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0.05)',
        xaxis_title='Word',
        yaxis_title='Frequency',
        showlegend=False,
        width=800,
        height=400
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    # Display chart
    st.plotly_chart(fig)

# effectiveness types frequency per discourse type (3 types of effectiveness and 7 types of discourse type)

## ---- Effectiveness frequency analysis by discourse type ---- ##

st.subheader("Most common effectiveness per Discourse Type")

# Create effectiveness frequency charts for each discourse type
for i, dtype in enumerate(discourseTypes):
    # Filter data for current discourse type
    mask = data['discourse_type'] == dtype
    effectiveness_counts = data[mask]['discourse_effectiveness'].value_counts().reset_index()
    effectiveness_counts.columns = ['Effectiveness', 'Frequency']

    # Create bar chart
    fig = px.bar(
        effectiveness_counts,
        x='Effectiveness',
        y='Frequency',
        title=f'Effectiveness for {dtype}',
        color_discrete_sequence=['#CDECAC']  # Using light green color from palette
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0.05)',
        xaxis_title='Effectiveness',
        yaxis_title='Frequency',
        showlegend=False,
        width=800,
        height=400
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    # Display chart
    st.plotly_chart(fig)
