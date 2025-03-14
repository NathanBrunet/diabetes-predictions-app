import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🎨 Custom Streamlit Theme
st.set_page_config(page_title="Diabetes Data Explorer", page_icon="📊", layout="wide")

# 🌟 Title with color
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎈 Diabetes Data Explorer</h1>", unsafe_allow_html=True)
st.info("Explore the TAIPEI_diabetes dataset with interactive tables and visualizations! 📊✨")

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/main/TAIPEI_diabetes.csv")

# Expandable section for raw data
with st.expander("📂 **View Dataset**"):
    st.write("### 📊 **Raw Data**")  
    st.dataframe(df)  

    # Define X and y
    st.write("### 🔍 **Feature Variables (X)**")
    X = df.drop("Diabetic", axis=1)
    st.dataframe(X)  

    st.write("### 🎯 **Target Variable (y: Diabetic)**")
    y = df["Diabetic"]
    st.write(y.value_counts())  # Shows class distribution

# 📊 **Data Visualization**
st.subheader("📈 Data Visualization")

# 🔹 Count plot of Diabetic variable
st.write("### 📊 Diabetes Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Diabetic"], palette="pastel", ax=ax)
st.pyplot(fig)

# 🔹 Boxplot for numerical features
st.write("### 📊 Boxplot of Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, palette="coolwarm", ax=ax)
st.pyplot(fig)

# 🔹 Correlation Heatmap
st.write("### 🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)





    

    



    
    
