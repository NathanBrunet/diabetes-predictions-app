import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 🎨 Streamlit Page Configuration
st.set_page_config(page_title="Diabetes Data Explorer", page_icon="📊", layout="wide")

# 🌟 Title and App Info
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎈 Diabetes Data Explorer</h1>", unsafe_allow_html=True)
st.info("Analyze and visualize the TAIPEI_diabetes dataset to predict diabetes outcomes based on various factors!")

# 📂 Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/main/TAIPEI_diabetes.csv")

# Drop 'PatientID' column if it exists
if 'PatientID' in df.columns:
    df.drop(columns=['PatientID'], inplace=True)

# 🚀 **Model Building and Prediction**
X = df.drop("Diabetic", axis=1)
y = df["Diabetic"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 📂 Sidebar for User Input
st.sidebar.header("User Input for Prediction")
user_data = {}
features = ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"]
limits = [(0, 20), (50, 250), (40, 200), (10, 100), (0, 1000), (10.0, 60.0), (0.0, 2.5), (18, 120)]

for feature, (min_val, max_val) in zip(features, limits):
    user_data[feature] = st.sidebar.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2)

# Prepare and scale user input
user_df = pd.DataFrame([user_data])
user_df_scaled = scaler.transform(user_df)

# Make Prediction
prediction = model.predict(user_df_scaled)

# 📊 **Data Visualization Section**
st.subheader("📈 Data Visualizations")

# **1. Diabetes Count Plot**
st.write("### 📊 Diabetes Cases")
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=df["Diabetic"], palette=["#1E88E5", "#D81B60"], ax=ax)
ax.set_xticklabels(["No Diabetes", "Diabetes"])
ax.set_ylabel("Count")
ax.set_xlabel("Diabetic Status")
ax.set_title("Diabetes Cases", fontsize=14)
st.pyplot(fig)

# **2. Improved Correlation Heatmap**
st.write("### 🔥 Feature Correlations")
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr(method='spearman')
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", mask=mask, vmin=-1, vmax=1, square=True, cbar=True)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
st.pyplot(fig)

# **Prediction Result**
with st.sidebar.expander("Prediction Result"):
    if prediction == 1:
        st.markdown(f"<h3 style='color: #D81B60; text-align: center;'>🚨 Prediction: The patient is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: #1E88E5; text-align: center;'>✅ Prediction: The patient is likely to not have diabetes.</h3>", unsafe_allow_html=True)
