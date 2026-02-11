import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# =============================
# Page Configuration
# =============================
st.set_page_config(page_title="NYC Taxi Hotspot Detection", layout="wide")

st.title("🚕 NYC Taxi Pickup Hotspot Detection")
st.markdown("Using **DBSCAN Clustering** to discover natural demand hotspots.")

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("⚙️ Model Settings")
eps_value = st.sidebar.slider("Select eps value", 0.1, 1.0, 0.3, 0.1)
min_samples = st.sidebar.slider("Select min_samples", 3, 20, 5)
row_limit = st.sidebar.slider("Rows to analyze", 1000, 50000, 15000, 1000)

# =============================
# File Upload Section
# =============================
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload NewYorkCityTaxiTripDuration.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue.")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded_file)

st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# =============================
# Column Validation
# =============================
required_columns = ['pickup_latitude', 'pickup_longitude']

if not all(col in df.columns for col in required_columns):
    st.error("CSV must contain 'pickup_latitude' and 'pickup_longitude' columns.")
    st.stop()

# Remove missing values
df = df.dropna(subset=required_columns)

# =============================
# Feature Selection
# =============================
X = df[['pickup_latitude', 'pickup_longitude']].head(row_limit)

if len(X) < 10:
    st.error("Not enough data points for clustering.")
    st.stop()

# =============================
# Scaling
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# DBSCAN Clustering
# =============================
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

# =============================
# Cluster Evaluation
# =============================
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_ratio = n_noise / len(labels)

st.subheader("📊 Cluster Evaluation")

col1, col2, col3 = st.columns(3)
col1.metric("Clusters", n_clusters)
col2.metric("Noise Points", n_noise)
col3.metric("Noise Ratio", round(noise_ratio, 4))

# =============================
# Silhouette Score
# =============================
mask = labels != -1

if len(set(labels[mask])) > 1:
    score = silhouette_score(X_scaled[mask], labels[mask])
    st.success(f"Silhouette Score: {round(score, 4)}")
else:
    st.warning("Silhouette Score: Not Applicable")

# =============================
# Visualization
# =============================
st.subheader("📍 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8,6))

for label in unique_labels:
    subset = X[labels == label]
    
    if label == -1:
        ax.scatter(
            subset['pickup_longitude'],
            subset['pickup_latitude'],
            color='black',
            marker='x',
            s=15,
            label='Noise'
        )
    else:
        ax.scatter(
            subset['pickup_longitude'],
            subset['pickup_latitude'],
            s=10,
            label=f'Cluster {label}'
        )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)

# =============================
# Automatic eps Comparison
# =============================
st.subheader("🔍 Automatic eps Comparison")

eps_values = [0.2, 0.3, 0.5]
results = {}

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    lab = db.fit_predict(X_scaled)
    
    mask = lab != -1
    
    if len(set(lab[mask])) > 1:
        score = silhouette_score(X_scaled[mask], lab[mask])
        results[eps] = score

if results:
    best_eps = max(results, key=results.get)
    st.success(f"Best eps value (based on silhouette): {best_eps}")
else:
    st.warning("No valid silhouette score found.")
