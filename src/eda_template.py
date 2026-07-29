# Week 2: Exploratory Data Analysis and Visualization
# Project: Customer Churn Prediction

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# Step 2: Load Dataset
# df = pd.read_csv('data/customer_data.csv')
print("Dataset Loaded Successfully")

# Step 3: Data Cleaning
def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# Step 4: Exploratory Data Analysis
def perform_eda(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # Correlation Heatmap
    plt.figure()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("diagrams/correlation.png")
    
    # Churn Distribution
    plt.figure()
    sns.countplot(x='Churn', data=df)
    plt.title("Customer Churn Distribution")
    plt.savefig("diagrams/churn_dist.png")

# Step 5: Feature Engineering
def feature_engineering(df):
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    return df

# Main Function
if __name__ == "__main__":
    # df = pd.read_csv('data/customer_data.csv')
    # df = clean_data(df)
    # df = feature_engineering(df)
    # perform_eda(df)
    print("EDA Template Ready. Uncomment lines when you have dataset.")
