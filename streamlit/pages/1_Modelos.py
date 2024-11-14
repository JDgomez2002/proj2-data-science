import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from pathlib import Path

path = "/Users/jdgomez/UVG/data-science/proj2-data-science"

# Set page title
st.set_page_config(page_title="Discourse Effectiveness Classifier", layout="wide")
st.title('Discourse Effectiveness Classifier')

# Model definitions
models = {
    'RoBERTa Base': 'roberta-base2',
    # 'RoBERTa Large': 'roberta-large',
    # 'BERT Base': 'bert-base'
}


# Initialize tokenizer
@st.cache_resource
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(f"{path}/trainedModels/{model_name}")


# Load the trained models and predictions
@st.cache_resource
def load_model_and_predictions(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(f'{path}/trainedModels/{model_name}')
        trainer = Trainer(model=model)
        with open(f'{path}/trainedModels/{model_name}-predictions.pkl', 'rb') as f:
            predictions = pickle.load(f)
        return trainer, predictions
    except Exception as e:
        return None, None


# Sidebar
st.sidebar.title('Model Selection')

# Model selection
selected_models = st.sidebar.multiselect(
    "Select models to compare",
    options=list(models.keys()),
    default=[list(models.keys())[0]]
)

# Input for text classification
input_text = st.sidebar.text_area(
    label="Classify discourse",
    placeholder="Enter discourse text to classify...",
    height=250
)

# Class labels
labels = ['Effective', 'Adequate', 'Ineffective']

# Main content
if selected_models:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Visualizations", "Classification Results"])

    with tab1:
        st.header("Model Performance Comparison")

        # Create metrics comparison table
        metrics_data = []

        for model_name in selected_models:
            trainer, predictions = load_model_and_predictions(models[model_name])
            if trainer and predictions:
                metrics = predictions[2]
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['test_accuracy']:.3f}",
                    'Precision': f"{metrics['test_precision']:.3f}",
                    'Recall': f"{metrics['test_recall']:.3f}",
                    'F1 Score': f"{metrics['test_f1']:.3f}"
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df.set_index('Model'))

    with tab2:
        # Confusion Matrices
        st.header("Confusion Matrices")
        cols = st.columns(len(selected_models))

        for idx, (col, model_name) in enumerate(zip(cols, selected_models)):
            trainer, predictions = load_model_and_predictions(models[model_name])
            if trainer and predictions:
                with col:
                    st.subheader(model_name)
                    true_labels = predictions[1]
                    pred_labels = np.argmax(predictions[0], axis=1)
                    cm = confusion_matrix(true_labels, pred_labels)

                    fig_cm = px.imshow(cm,
                                       labels=dict(x="Predicted", y="True", color="Count"),
                                       x=labels,
                                       y=labels,
                                       aspect="auto",
                                       color_continuous_scale="Blues",
                                       title=model_name)
                    st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curves
        st.header("ROC Curves")
        fig_roc = go.Figure()

        for model_name in selected_models:
            trainer, predictions = load_model_and_predictions(models[model_name])
            if trainer and predictions:
                probabilities = predictions[0]
                true_labels = predictions[1]

                for i, label in enumerate(labels):
                    fpr, tpr, _ = roc_curve(true_labels == i, probabilities[:, i])
                    roc_auc = auc(fpr, tpr)

                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'{model_name} - {label} (AUC = {roc_auc:.2f})',
                        mode='lines'
                    ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))

        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=800,
            height=600
        )
        st.plotly_chart(fig_roc)

    with tab3:
        if input_text:
            st.header("Classification Results")

            # Create columns for each model's predictions
            cols = st.columns(len(selected_models))

            for idx, (col, model_name) in enumerate(zip(cols, selected_models)):
                trainer, predictions = load_model_and_predictions(models[model_name])
                tokenizer = load_tokenizer(models[model_name])

                if trainer:
                    with col:
                        st.subheader(model_name)

                        # Check if MPS is available
                        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

                        # Move model to the appropriate device
                        trainer.model.to(device)

                        # Tokenize input text
                        inputs = tokenizer(input_text,
                                           padding='max_length',
                                           truncation=True,
                                           max_length=128,
                                           return_tensors="pt")
                        
                        inputs = {key: value.to(device) for key, value in inputs.items()}

                        # Get prediction
                        with torch.no_grad():
                            outputs = trainer.model(**inputs)
                            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            predicted_class = torch.argmax(predictions).item()

                        # Display results
                        probabilities = predictions[0].cpu().numpy()

                        st.write("**Predicted Class:**")
                        st.write(f"**{labels[predicted_class]}**")

                        st.write("**Class Probabilities:**")
                        for class_name, prob in zip(labels, probabilities):
                            st.write(f"{class_name}: {prob * 100:.2f}%")
else:
    st.warning("Please select at least one model to compare.")