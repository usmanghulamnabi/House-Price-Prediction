# streamlit_house_price_clean.py

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(
    page_title="House Price Predictor 🏡",
    page_icon="🏠",
    layout="wide"
)

# --- Header ---
st.markdown("""
<h1 style='text-align: center; color: #4B0082;'>House Price Prediction App 🏠</h1>
<p style='text-align: center; color: #6E6E6E;'>Cleaned features, robust predictions</p>
""", unsafe_allow_html=True)

# --- Load Dataset ---
data_path = "data.csv"  # Replace with your dataset path
data = pd.read_csv(data_path)

st.markdown("### 📊 Dataset Preview")
st.dataframe(data.head())

# --- Clean Numeric Features ---
numeric_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                'waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated','price']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# remove unreliably signed prices and outliers from target distribution
data = data[data['price'] > 0]
q1, q99 = data['price'].quantile([0.01, 0.99])
data = data[(data['price'] >= q1) & (data['price'] <= q99)]

data = data.dropna(subset=numeric_cols)

# --- Feature Engineering ---
current_year = 2026
if 'date' in data.columns:
    # if date exists, extract year from it
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])
    current_year = data['date'].dt.year.max()

data['house_age'] = current_year - data['yr_built']
data['renovated_age'] = np.where(data['yr_renovated'] > 0,
                                  current_year - data['yr_renovated'],
                                  data['house_age'])

data['house_age'] = data['house_age'].clip(lower=0)
data['renovated_age'] = data['renovated_age'].clip(lower=0)

numeric_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                    'waterfront','view','condition','sqft_above','sqft_basement',
                    'house_age','renovated_age']

# --- Low-Cardinality Categorical Features ---
categorical_features = ['city','country']
for col in categorical_features:
    if col not in data.columns:
        data[col] = 'unknown'

# Keep categorical values for LightGBM instead of manual one-hot encoding
data[categorical_features] = data[categorical_features].astype('category')

# --- Prepare training data ---
X = data[numeric_features + categorical_features].copy()
y = np.log1p(data['price'])  # log-target regression

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train LightGBM ---
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)
fit_kwargs = {
    'eval_set': [(X_test, y_test)],
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50,
    'categorical_feature': categorical_features,
    'verbose': 50
}

try:
    model.fit(X_train, y_train, **fit_kwargs)
except TypeError as e:
    # Handle older LightGBM API versions where early_stopping_rounds/verbose not accepted
    err_msg = str(e)
    if 'early_stopping_rounds' in err_msg or 'verbose' in err_msg or 'eval_set' in err_msg:
        fit_kwargs.pop('early_stopping_rounds', None)
        fit_kwargs.pop('verbose', None)
        model.fit(X_train, y_train, **fit_kwargs)
    else:
        raise
# --- Evaluate ---
y_pred_log = model.predict(X_test)
y_test_log = y_test

# Convert back from log scale for final metrics
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_log)

r2 = r2_score(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

st.markdown("### ⚡ Model Performance")
st.markdown(f"**R² Score:** {r2:.3f}  \n**RMSE:** {rmse:,.2f}")
st.markdown("---")
st.markdown("### 🏡 Predict Price for a New House")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0, 10, 2)
    sqft_living = st.number_input("Sqft Living", 0, 10000, 1500)
    sqft_lot = st.number_input("Sqft Lot", 0, 100000, 5000)
    floors = st.number_input("Floors", 1, 5, 1)
    waterfront = st.selectbox("Waterfront", [0,1])
    view = st.slider("View Quality", 0, 4, 0)

with col2:
    condition = st.slider("Condition", 1, 5, 3)
    sqft_above = st.number_input("Sqft Above", 0, 10000, 1500)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
    yr_built = st.number_input("Year Built", 1800, 2026, 2000)
    yr_renovated = st.number_input("Year Renovated", 0, 2026, 0)
    city = st.text_input("City", "Seattle")
    country = st.text_input("Country", "USA")

if st.button("💰 Predict Price"):
    house_age = 2026 - yr_built
    renovated_age = 2026 - yr_renovated if yr_renovated>0 else house_age

    input_df = pd.DataFrame({
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'sqft_living':[sqft_living],
        'sqft_lot':[sqft_lot],
        'floors':[floors],
        'waterfront':[waterfront],
        'view':[view],
        'condition':[condition],
        'sqft_above':[sqft_above],
        'sqft_basement':[sqft_basement],
        'house_age':[house_age],
        'renovated_age':[renovated_age],
        'city':[city],
        'country':[country]
    })

    # Format categorical features for the model
    input_df[categorical_features] = input_df[categorical_features].astype('category')
    X_input = input_df[numeric_features + categorical_features].copy()

    predicted_log_price = model.predict(X_input)
    predicted_price = np.expm1(predicted_log_price)
    st.success(f"🏷 Predicted House Price: **${predicted_price[0]:,.2f}**")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6E6E6E;'>Made by Dr. Muhammad Usman | "
    "<a href='https://github.com/YourGitHubUsername' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)