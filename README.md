**House Price Prediction App 🏡**

A modern machine learning web application built with Python, LightGBM, and Streamlit to predict house prices based on key property features.

This project demonstrates clean data preprocessing, feature engineering, and an interactive user interface for predicting real estate prices in real time.

**Features**

Predict house prices using a robust LightGBM regression model.

Supports both numeric and categorical features (bedrooms, bathrooms, sqft_living, floors, city, country, etc.).

Automatic feature engineering: calculates house_age and renovated_age from the year built and renovated.

Clean and modern Streamlit UI with two-column layout for input and instant predictions.

Responsive and interactive interface: easy for end users to input features and get predictions.

**How it Works**

Loads your dataset (data.csv) containing features like bedrooms, bathrooms, square footage, floors, view, condition, and location.

Cleans numeric features, handles missing values, and removes noisy columns.

One-hot encodes low-cardinality categorical variables (city, country).

Trains a LightGBM regression model on the clean data.

Provides a Streamlit UI for user input and real-time house price prediction.

**Requirements**

Python 3.8+

pandas

numpy

scikit-learn

lightgbm

streamlit

**Author**

**Dr. Muhammad Usman**
