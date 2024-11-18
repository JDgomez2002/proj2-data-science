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

# Set page title
st.set_page_config(page_title="Discourse Effectiveness Classifier", layout="wide")
st.title('Discourse Effectiveness Classifier')

# Model definitions
models = {
    'RoBERTa Base': 'roberta-base',
    'deBERTa Base': 'deberta-base',
    'Xlnet Base': 'xlnet-base',
    'Electra Base': 'electra-base',
    'Bert Base': 'bert-base'
}

custom_color_scale = [
    [0.0, "#000000"],  # Black
    [0.2, "#E5E5E5"],  # Light gray
    [0.4, "#FFFFFF"],  # White
    [0.6, "#CDECAC"],  # Light green
    [0.8, "#2D9494"],  # Teal
    [1.0, "#CC5A49"]   # Reddish orange
]

color_map = {
    'Training': '#E5E5E5',    
    'Validation': '#CC5A49'   # Turquoise
}


# Initialize tokenizer
@st.cache_resource
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(f"trainedModels/{model_name}")


# Load the trained models and predictions
@st.cache_resource
def load_model_and_predictions(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(f'trainedModels/{model_name}')
        trainer = Trainer(model=model)
        url = f'trainedModels/{model_name}/{model_name}-predictions.pkl'
        with open(url, 'rb') as f:
            predictions = pickle.load(f)
        return trainer, predictions
    except Exception as e:
        print("Error loading model and predictions:", model_name, e)
        return None, None


def load_model_history(model_name):
    try:
        url = f'trainedModels/{model_name}/{model_name}.pkl'
        with open(url, 'rb') as f:
            history = pickle.load(f)

        train_losses = []
        eval_losses = []
        epochs_loss = []
        epochs = []

        for log in history:
            if 'loss' in log and 'epoch' in log:
                train_losses.append(log['loss'])
                epochs_loss.append(log['epoch'])
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
            if 'eval_accuracy' in log:
                epochs.append(log['epoch'])
        
        return train_losses, eval_losses, epochs_loss, epochs
    except Exception as e:
        print("Error loading model, predictions, and history:", model_name, e)
        return None, None, None

# Sidebar
st.sidebar.title('Model Selection')

# Model selection
selected_models = st.sidebar.multiselect(
    "Select models to compare",
    options=list(models.keys()),
    default=list(models.keys())
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
                    'Accuracy': f"{metrics['test_accuracy'] * 100:.2f}%",
                    'Precision': f"{metrics['test_precision'] * 100:.2f}%",
                    'Recall': f"{metrics['test_recall'] * 100:.2f}%",
                    'F1 Score': f"{metrics['test_f1'] * 100:.2f}%"
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df.set_index('Model'))

    with tab2:
        # Confusion Matrices
        st.header("Confusion Matrices")
        col1, col2 = st.columns(2)

        for idx, model_name in enumerate(selected_models):
            trainer, predictions = load_model_and_predictions(models[model_name])
            if trainer and predictions:
                true_labels = predictions[1]
                pred_labels = np.argmax(predictions[0], axis=1)
                cm = confusion_matrix(true_labels, pred_labels)

                with col1 if idx % 2 == 0 else col2:
                    st.subheader(model_name)
                    fig_cm = px.imshow(cm,
                        labels=dict(x="Predicted", y="True", color="Count"),
                        x=labels,
                        y=labels,
                        color_continuous_scale=custom_color_scale,
                        text_auto=True,)
                    st.plotly_chart(fig_cm, use_container_width=True, key=idx)

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


        # Training and Validation Loss
        st.header("Training and Validation Loss")
        col3, col4 = st.columns(2)
        fig_history = go.Figure()

        for idx, model_name in enumerate(selected_models):
            train_losses, eval_losses, epochs_loss, epochs = load_model_history(models[model_name])
            if train_losses and eval_losses and epochs_loss:
                # Crear DataFrame con los datos
                df = pd.DataFrame({
                    'Epoch': epochs_loss + epochs,  # Combinar epochs para ambas series
                    'Loss': train_losses + eval_losses,  # Combinar valores de loss
                    'Type': ['Training']*len(epochs_loss) + ['Validation']*len(epochs)  # Identificar tipo
                })

                # Crear gráfica con Plotly Express
                fig_history = px.line(df, 
                            x='Epoch', 
                            y='Loss',
                            color='Type',
                            markers=True,
                            title=f'{model_name}',
                            color_discrete_map=color_map)

                # Personalizar la gráfica
                fig_history.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title=None,
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1)
                )

                # Mostrar en Streamlit
                with col3 if idx % 2 == 0 else col4:
                    st.plotly_chart(fig_history, use_container_width=True)

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
            st.warning("Please enter a text to classify.")
else:
    st.warning("Please select at least one model to compare.")