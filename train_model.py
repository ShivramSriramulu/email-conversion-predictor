import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv("data 2.csv")

# One-hot encode categorical variables
categorical_cols = ['zip_code', 'channel', 'offer']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split features and target
X = df_encoded.drop('conversion', axis=1)
y = df_encoded['conversion']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model, scaler, and feature names
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickl