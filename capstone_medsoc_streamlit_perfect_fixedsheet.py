import streamlit as st
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

st.title("Capstone Medsoc - Prediksi Engagement & Rekomendasi Jam Posting")

# Link Google Drive (gunakan file ID Anda yang sudah shareable publik)
file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'dataset_engagement.xlsx'

# Download otomatis (cek jika sudah ada tidak akan download ulang)
try:
    gdown.download(url, output, quiet=False)
    st.success("Dataset berhasil diunduh dari Google Drive!")
except Exception as e:
    st.error(f"Error saat mengunduh file: {e}")
    st.stop()

# Load data dari file yang sudah diunduh
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

st.write("Data setelah preprocessing:", df.head())

# Modeling preparation
features = ['Time Periods', 'Weekday Type', 'Engagement Rate', 'Total Engagement',
            'Platform', 'Post Type', 'Sentiment', 'Age Group', 'Audience Gender', 'Hour']

df_model = df[features].copy()
for col in ['Weekday Type', 'Time Periods', 'Sentiment', 'Age Group', 'Audience Gender']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

df_model = pd.get_dummies(df_model, columns=['Platform', 'Post Type'], drop_first=True)

X = df_model.drop(['Total Engagement'], axis=1)
y = df_model['Total Engagement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.write(f"Model Evaluation: MAE={mean_absolute_error(y_test, y_pred):.2f}, MSE={mean_squared_error(y_test, y_pred):.2f}")

# Cari jam terbaik
best_hour = df.groupby('Hour')['Total Engagement'].mean().idxmax()

st.header("Prediksi Engagement Berdasarkan Input Anda")

# User input interaktif
platform_input = st.selectbox("Pilih Platform", df['Platform'].unique())
post_type_input = st.selectbox("Pilih Jenis Post", df['Post Type'].unique())
gender_input = st.selectbox("Pilih Gender Target", df['Audience Gender'].unique())

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
input_df = pd.get_dummies(input_df, columns=['Platform', 'Post Type'], drop_first=True)
input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

if input_df.isnull().values.any() or input_df.empty:
    st.error("Input tidak valid atau kosong.")
else:
    pred_engagement = model.predict(input_df)
    st.success(f"Prediksi Total Engagement : {pred_engagement[0]:.2f}")
    st.info(f"Rekomendasi Jam Terbaik : {best_hour}.00 WIB")
