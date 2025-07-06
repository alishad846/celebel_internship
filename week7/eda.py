import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(layout="wide", page_title="Wine Dataset EDA")

# Load cleaned dataset
df = pd.read_parquet("cleaned_wine.parquet")

st.title("🍷 Wine Dataset - Visual Analysis")

# 1. Show class distribution
st.subheader("Class Distribution")
st.bar_chart(df["class"].value_counts().sort_index())

# 2. Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
corr = df.drop("class", axis=1).corr()
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# 3. Feature Distribution
st.subheader("Distribution of Features")
feature = st.selectbox("Select a feature", df.columns[:-1])
fig2, ax2 = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax2)
st.pyplot(fig2)

# 4. PCA Scatterplot (2D projection)
st.subheader("PCA Projection")
X = df.drop("class", axis=1)
y = df["class"]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["class"] = y

fig3, ax3 = plt.subplots()
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class", palette="Set1", ax=ax3)
st.pyplot(fig3)
