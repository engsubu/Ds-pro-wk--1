import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction - Week 2 EDA")
st.markdown("### Virtual Data Science Explorer Internship")

uploaded_file = st.file_uploader("Upload customer_data.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("2. Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("3. Data Info")
    st.write(df.describe())

    st.subheader("4. Visualizations")

    if 'Churn' in df.columns:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax1, palette="Set2")
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    st.success("EDA Complete! Ready for Week 3 Modeling")
else:
    st.info("👆 Please upload a CSV file to start EDA")
