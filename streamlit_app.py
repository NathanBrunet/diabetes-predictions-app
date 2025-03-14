import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🎨 Set Streamlit Page Configuration
st.set_page_config(page_title="Diabetes Data Explorer", page_icon="📊", layout="wide")

# 🌟 Title with color
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎈 Diabetes Data Explorer</h1>", unsafe_allow_html=True)
st.info("Analyze and visualize the TAIPEI_diabetes dataset to predict diabetes outcomes based on various factors!")

# 📂 Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/main/TAIPEI_diabetes.csv")

# 🔍 Show Raw Data
with st.expander("📂 **View Dataset**"):
    st.write("### Raw Data")  
    st.dataframe(df)

# 📊 **Data Visualization Section**
st.subheader("📈 Data Visualization")

# 🔹 Custom Color Palette
colors = ["#FF4B4B", "#1E88E5", "#FFC107", "#2E7D32", "#D81B60", "#8E24AA"]

#  **1. Diabetes Count Plot (Better Colors)**
st.write("###  Diabetes Cases")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Diabetic"], palette=["#1E88E5", "#D81B60"], ax=ax)
ax.set_xticklabels(["No Diabetes", "Diabetes"])
ax.set_ylabel("Count")
ax.set_xlabel("Diabetic Status")
st.pyplot(fig)

# 📊 **2. Age Distribution (More Informative)**
st.write("### 🎂 Age Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True, color="#FF4B4B", edgecolor="black")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

# 🔍 **3. Glucose Levels vs. Age (Scatter Plot)**
st.write("### 🍬 Plasma Glucose vs. Age")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df["Age"], y=df["PlasmaGlucose"], hue=df["Diabetic"], palette=["#1E88E5", "#D81B60"], alpha=0.7)
ax.set_xlabel("Age")
ax.set_ylabel("Plasma Glucose Level")
st.pyplot(fig)

#  **4. Correlation Heatmap (Improved Style)**
st.write("### 🔥 Feature Correlations")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
st.pyplot(fig)

# 🚀 **Model Building and Prediction**

# Split data into features (X) and target (y)
X = df.drop("Diabetic", axis=1)
y = df["Diabetic"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### 📈 Model Accuracy: {accuracy * 100:.2f}%")
st.write("### 📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.write("### 📋 Classification Report:")
st.write(classification_report(y_test, y_pred))

# 📋 **User Input for Prediction**
st.subheader("📝 Enter Patient Data for Prediction")

age = st.number_input("Age", min_value=18, max_value=120)
pregnancies = st.number_input("Number of Pregnancies", min_value=0)
plasma_glucose = st.number_input("Plasma Glucose", min_value=50, max_value=250)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=200)
triceps_thickness = st.number_input("Triceps Skin Fold Thickness", min_value=10, max_value=100)
serum_insulin = st.number_input("Serum Insulin", min_value=0, max_value=1000)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)

# Prepare the input for prediction
user_input = [[pregnancies, plasma_glucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age]]
user_input_scaled = scaler.transform(user_input)  # Standardize input

# Make prediction
prediction = model.predict(user_input_scaled)

if prediction == 1:
    st.write("### 🚨 Prediction: The patient is likely to have diabetes.")
else:
    st.write("### ✅ Prediction: The patient is likely to not have diabetes.")







    

    



    
    
