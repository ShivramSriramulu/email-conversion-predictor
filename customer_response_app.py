import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

# Load model, scaler, feature names
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data 2.csv")

df = load_data()

# Sidebar inputs
st.sidebar.header("User Input")
def user_input_features():
    recency = st.sidebar.slider("Recency", 0, 30, 10)
    history = st.sidebar.slider("History", 0.0, 1000.0, 200.0)
    used_discount = st.sidebar.selectbox("Used Discount", [0, 1])
    used_bogo = st.sidebar.selectbox("Used BOGO", [0, 1])
    is_referral = st.sidebar.selectbox("Is Referral", [0, 1])
    zip_code = st.sidebar.selectbox("Zip Code", df["zip_code"].unique())
    channel = st.sidebar.selectbox("Channel", df["channel"].unique())
    offer = st.sidebar.selectbox("Offer", df["offer"].unique())

    return pd.DataFrame([{
        "recency": recency,
        "history": history,
        "used_discount": used_discount,
        "used_bogo": used_bogo,
        "is_referral": is_referral,
        "zip_code": zip_code,
        "channel": channel,
        "offer": offer
    }])

input_df = user_input_features()

# Combine and encode input with reference data
df_temp = df.drop(columns=["conversion"])
df_combined = pd.concat([input_df, df_temp], axis=0)
df_encoded = pd.get_dummies(df_combined)
df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

# Scale and predict
X_input = scaler.transform(df_encoded.iloc[0:1])
prob = model.predict_proba(X_input)[0][1]
st.metric(label="Predicted Conversion Probability", value=f"{prob*100:.2f}%")

# SHAP explanation
# SHAP Bar Plot (wrapped in plt.figure)
st.subheader("🔍 SHAP Feature Importance (Local Explanation)")
explainer = shap.Explainer(model)
shap_values = explainer(df_encoded.iloc[0:1])

# Create a matplotlib-compatible plot
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values[0], show=False)
st.pyplot(plt.gcf())  # Use current matplotlib figure


# Segment Clusters
st.subheader("📊 Customer Segments (KMeans)")
seg_cols = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
scaler_seg = MinMaxScaler()
X_seg = scaler_seg.fit_transform(df[seg_cols])
df["Segment"] = KMeans(n_clusters=4, random_state=42).fit_predict(X_seg)

fig_seg = plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Segment")
plt.title("Segment Distribution")
st.pyplot(fig_seg)

# Evaluation block — use only original dataset (exclude input row)
df_eval = df_encoded.iloc[1:]  # Skip first row (user input)
X_all = scaler.transform(df_eval)
y_all = df["conversion"]  # 64000 labels

# Get predictions
y_pred = model.predict(X_all)
y_proba = model.predict_proba(X_all)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_all, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], "k--")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig_roc)

# Confusion Matrix
cm = confusion_matrix(y_all, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)
