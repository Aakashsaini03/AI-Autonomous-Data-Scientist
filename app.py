import streamlit as st
from pipeline import load_data, data_summary, clean_data
from model import train_model
from utils import plot_data
from pipeline import load_data
from utils import plot_data
from model import train_model
from pipeline import data_summary, clean_data
from model import train_model
from utils import plot_data
from llm import generate_insights_llm

st.title("🤖 AI Autonomous Data Scientist")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Summary
    summary = data_summary(df)
    st.subheader("Data Summary")
    st.write(summary)

    # Cleaning
    df = clean_data(df)

    # Visualization
    st.subheader("EDA")
    plots = plot_data(df)
    for fig in plots:
        st.pyplot(fig)

    # Target selection
    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        model, score, metric = train_model(df, target)

        st.subheader("Model Performance")
        st.write(f"{metric}: {score}")

        # Insights
        insights = f"""
        Dataset processed successfully.
        Model achieved {metric}: {score}
        """
        st.subheader("AI Insights")
        st.write(insights)