import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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







    

    



    
    
