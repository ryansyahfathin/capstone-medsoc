
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.title("Capstone Medsoc - Analisis Engagement & Prediksi Waktu Posting Terbaik")

uploaded_file = st.file_uploader("Upload dataset Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Working File')
    st.success("Dataset berhasil diupload dan dibaca!")
    
    df['Time'] = df['Post Timestamp'].dt.time
    df['Date'] = df['Post Timestamp'].dt.date
    df['Day'] = df['Post Timestamp'].dt.day
    df['Month'] = df['Post Timestamp'].dt.month
    df['Year'] = df['Post Timestamp'].dt.year
    
    Reorder = ['Time', 'Date', 'Day', 'Month', 'Year', 'Weekday Type', 'Time Periods', 'Platform',
               'Post Type', 'Post Content', 'Likes', 'Comments', 'Shares', 'Impressions', 'Reach',
               'Engagement Rate', 'Audience Age', 'Age Group', 'Audience Gender', 'Audience Location',
               'Audience Continent', 'Audience Interests', 'Sentiment']
    df = df[Reorder]
    
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Total Engagement'] = df[['Likes', 'Comments', 'Shares']].sum(axis=1)
    
    st.write("Preview data:", df.head())

    features = ['Time Periods', 'Weekday Type', 'Engagement Rate', 'Total Engagement',
                'Platform', 'Post Type', 'Sentiment', 'Age Group', 'Audience Gender', 'Hour']
    df_encoded = df[features].copy()
    le = LabelEncoder()
    for col in ['Weekday Type', 'Time Periods', 'Sentiment', 'Age Group', 'Audience Gender']:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    df_encoded = pd.get_dummies(df_encoded, columns=['Platform', 'Post Type'], drop_first=True)

    X = df_encoded.drop(['Total Engagement'], axis=1)
    y = df_encoded['Total Engagement']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)

    st.write(f"**Gradient Boosting - MAE:** {mae_gb:.2f}")
    st.write(f"**Gradient Boosting - MSE:** {mse_gb:.2f}")

    st.header("Simulasi Prediksi Engagement")
    platform_input = st.selectbox("Pilih platform", df['Platform'].unique())
    post_type_input = st.selectbox("Pilih jenis post", df['Post Type'].unique())
    gender_input = st.selectbox("Pilih target gender", df['Audience Gender'].unique())
    age_group_input = st.selectbox("Pilih kelompok usia", df['Age Group'].unique())

    data_input = {
        'Platform': [platform_input],
        'Post Type': [post_type_input],
        'Audience Gender': [gender_input],
        'Age Group': [age_group_input],
        'Weekday Type': ['Weekday'],
        'Time Periods': ['Morning'],
        'Sentiment': ['Positive'],
        'Engagement Rate': [df['Engagement Rate'].mean()],
        'Hour': [12]
    }
    input_df = pd.DataFrame(data_input)
    input_df = pd.get_dummies(input_df, columns=['Platform', 'Post Type'], drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    predicted_engagement = gb_model.predict(input_df)

    st.subheader("Hasil Prediksi:")
    st.write(f"Prediksi Engagement Rate : **{predicted_engagement[0]:.2f}**")
    st.write(f"Rekomendasi posting pada jam : **12.00 WIB**")
else:
    st.info("Silahkan upload file terlebih dahulu.")
