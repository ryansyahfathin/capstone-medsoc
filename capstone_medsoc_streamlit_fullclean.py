
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.title("Capstone Medsoc - Analisis Engagement & Prediksi Waktu Posting Terbaik")

uploaded_file = st.file_uploader("Upload dataset Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Dataset berhasil diupload!")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Preprocessing wajib sebelum feature engineering
    df['Hour'] = pd.to_datetime(df['Post Timestamp']).dt.hour
    df['Total Engagement'] = df[['Likes', 'Comments', 'Shares']].sum(axis=1)

    # Cek kolom yang tersedia
    st.write("Kolom tersedia setelah preprocessing:", df.columns.tolist())

    # Definisikan fitur yang akan digunakan
    features = ['Time Periods', 'Weekday Type', 'Engagement Rate', 'Total Engagement',
                'Platform', 'Post Type', 'Sentiment', 'Age Group', 'Audience Gender', 'Hour']

    # Safety check fitur
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error(f"Kolom yang hilang atau belum diproses: {missing_features}")
        st.stop()

    # Lanjut ke encoding
    df_encoded = df[features].copy()
    le = LabelEncoder()
    for col in ['Weekday Type', 'Time Periods', 'Sentiment', 'Age Group', 'Audience Gender']:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    df_encoded = pd.get_dummies(df_encoded, columns=['Platform', 'Post Type'], drop_first=True)

    # Modeling
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

    # Cari jam terbaik berdasarkan data historis
    best_hour = df.groupby('Hour')['Total Engagement'].mean().idxmax()

    # User input
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
        'Hour': [best_hour]
    }
    input_df = pd.DataFrame(data_input)
    input_df = pd.get_dummies(input_df, columns=['Platform', 'Post Type'], drop_first=True)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    if input_df.empty or input_df.isnull().values.any():
        st.error("Input data tidak valid. Pastikan semua input terisi dengan benar.")
    else:
        predicted_engagement = gb_model.predict(input_df)
        st.subheader("Hasil Prediksi:")
        st.write(f"Prediksi Engagement Rate : **{predicted_engagement[0]:.2f}**")
        st.write(f"Rekomendasi posting pada jam : **{best_hour}.00 WIB**")

else:
    st.info("Silakan upload file dataset terlebih dahulu.")
