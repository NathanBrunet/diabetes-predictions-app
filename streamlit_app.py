import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

st.set_page_config(page_title="Diabetes Data Explorer", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎈 Diabetes Data Explorer</h1>", unsafe_allow_html=True)
st.info("Explore the TAIPEI_diabetes dataset with interactive visualizations! 📊✨")

df = pd.read_csv("https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/main/TAIPEI_diabetes.csv")


with st.expander("📂 **View Dataset**"):
    st.write("### 📊 **Raw Data**")  
    st.dataframe(df.style.set_properties(**{'background-color': '#FFF3E0', 'color': 'black'}))  

   
    st.write("### 🔍 **Feature Variables (X)**")
    X = df.drop("Diabetic", axis=1)
    st.dataframe(X.style.set_properties(**{'background-color': '#E0F7FA', 'color': 'black'}))  

    st.write("### 🎯 **Target Variable (y: Diabetic)**")
    y = df["Diabetic"]
    st.write(y.value_counts())  


st.markdown("<h2 style='text-align: center; color: #4B7BFF;'>📊 Data Visualizations</h2>", unsafe_allow_html=True)


st.write("### 🌈 Feature Distributions (Histograms)")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

colors = ["#FF6F61", "#6B4226", "#66B2FF", "#FFB733", "#8E44AD", "#28B463", "#FFC300", "#2E86C1"]  

for i, col in enumerate(X.columns):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color=colors[i])
    axes[i].set_title(col, fontsize=12, fontweight='bold', color=colors[i])

plt.tight_layout()
st.pyplot(fig)

st.write("### 🚨 Outlier Detection (Boxplots)")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    sns.boxplot(y=df[col], ax=axes[i], color=colors[i])
    axes[i].set_title(col, fontsize=12, fontweight='bold', color=colors[i])

plt.tight_layout()
st.pyplot(fig)


st.write("### 🔥 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=1, linecolor="black")
st.pyplot(fig)


st.write("### 📊 Diabetes vs. Non-Diabetes Count")
fig, ax = plt.subplots()
sns.countplot(x=df["Diabetic"], palette=["#FF6F61", "#66B2FF"], ax=ax)
ax.set_xticklabels(["No Diabetes", "Diabetes"])
ax.set_title("Diabetes Count", fontsize=14, fontweight="bold")
st.pyplot(fig)




    

    



    
    
