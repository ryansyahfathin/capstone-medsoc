
import streamlit as st
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

st.title("Capstone Medsoc - Prediksi Engagement (Fast Mode & Alternative Model)")

# Download dataset dari Google Drive
file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'dataset_engagement.xlsx'

gdown.download(url, output, quiet=False)
df = pd.read_excel(output, sheet_name='Working File')

# Preprocessing kolom wajib
df['Hour'] = pd.to_datetime(df['Post Timestamp']).dt.hour
df['Total Engagement'] = df[['Likes', 'Comments', 'Shares']].sum(axis=1)

# Kolom default yang ditambahkan jika hilang
default_columns = {
    'Time Periods': 'Morning',
    'Weekday Type': 'Weekday',
    'Age Group': 'Adolescent Adults'
}
for col, default in default_columns.items():
    if col not in df.columns:
        df[col] = default

# Definisi fitur
features = ['Time Periods', 'Weekday Type', 'Engagement Rate', 'Total Engagement',
            'Platform', 'Post Type', 'Sentiment', 'Age Group', 'Audience Gender', 'Hour']

df_model = df[features].copy()
for col in ['Weekday Type', 'Time Periods', 'Sentiment', 'Age Group', 'Audience Gender']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

df_model = pd.get_dummies(df_model, columns=['Platform', 'Post Type'], drop_first=True)
reference_columns = df_model.drop('Total Engagement', axis=1).columns

X = df_model.drop(['Total Engagement'], axis=1)
y = df_model['Total Engagement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Optimized Gradient Boosting
model_gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
model_gb.fit(X_train, y_train)

# Model 2: RandomForest
model_rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model_rf.fit(X_train, y_train)

# Cari jam terbaik
best_hour = df.groupby('Hour')['Total Engagement'].mean().idxmax()

st.header("Prediksi Engagement Berdasarkan Input Anda")
platform_input = st.selectbox("Pilih Platform", df['Platform'].unique())
post_type_input = st.selectbox("Pilih Jenis Post", df['Post Type'].unique())
gender_input = st.selectbox("Pilih Gender Target", df['Audience Gender'].unique())

if st.button("Predict Engagement"):
    data_input = {
        'Platform': [platform_input],
        'Post Type': [post_type_input],
        'Audience Gender': [gender_input],
        'Age Group': ['Adolescent Adults'],
        'Weekday Type': ['Weekday'],
        'Time Periods': ['Morning'],
        'Sentiment': ['Positive'],
        'Engagement Rate': [df['Engagement Rate'].mean()],
        'Hour': [best_hour]
    }
    input_df = pd.DataFrame(data_input)
    for col in ['Weekday Type', 'Time Periods', 'Sentiment', 'Age Group', 'Audience Gender']:
        input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])
    input_df = pd.get_dummies(input_df, columns=['Platform', 'Post Type'], drop_first=True)
    input_df = input_df.reindex(columns=reference_columns, fill_value=0)

    if input_df.isnull().values.any() or input_df.empty or input_df.shape[1] != X_train.shape[1]:
        st.error("Input data tidak valid atau kolom tidak sesuai model training.")
    else:
        # Gradient Boosting Prediction
        start_gb = time.time()
        pred_engagement_gb = model_gb.predict(input_df)
        time_gb = time.time() - start_gb

        # RandomForest Prediction
        start_rf = time.time()
        pred_engagement_rf = model_rf.predict(input_df)
        time_rf = time.time() - start_rf

        st.success(f"Gradient Boosting Prediksi: {pred_engagement_gb[0]:.2f} (waktu: {time_gb:.2f} detik)")
        st.info(f"RandomForest Prediksi: {pred_engagement_rf[0]:.2f} (waktu: {time_rf:.2f} detik)")
        st.info(f"Rekomendasi Jam Terbaik : {best_hour}.00 WIB")
