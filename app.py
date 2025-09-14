import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------
# Konfigurasi file lokal
# -----------------------------------------
MODEL_PATH = "model_final.keras" 
SCALER_PATH = "scaler.pkl"
DEFAULT_LOOKBACK = 60

st.set_page_config(page_title="Prediksi Harga LSTM", layout="wide")
st.title("Prediksi Harga LSTM")
st.info("Silakan unduh dataset dari GitHub (ikon kanan atas) sebelum melakukan upload di aplikasi ini.")

# -----------------------------------------
# Load model dan scaler otomatis
# -----------------------------------------
@st.cache_resource
def load_model(path: str):
    model = tf.keras.models.load_model(path, compile=False)
    return model

@st.cache_resource
def load_scaler(path: str):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, MinMaxScaler):
        st.warning("Objek yang dimuat bukan MinMaxScaler. Pastikan file scaler benar.")
    return scaler

try:
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    st.success("Model dan scaler berhasil dimuat otomatis.")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler. Detail: {e}")
    st.stop()

# -----------------------------------------
# Sidebar pengaturan
# -----------------------------------------
with st.sidebar:
    st.header("Pengaturan")
    lookback = st.number_input("Lookback", min_value=5, max_value=500, value=DEFAULT_LOOKBACK, step=1)
    n_future = st.number_input("Langkah ke depan", min_value=1, max_value=365, value=30, step=1)
    freq = st.selectbox("Frekuensi tanggal", options=["D", "B", "H"], index=0)
    tail_n = st.number_input("Tampilkan historis terakhir N baris", min_value=30, max_value=5000, value=300, step=10)

# -----------------------------------------
# Upload dataset
# -----------------------------------------
uploaded = st.file_uploader("Upload CSV dengan kolom Date dan Close", type=["csv"])
if not uploaded:
    st.info("Silakan upload dataset untuk mulai prediksi.")
    st.stop()

# -----------------------------------------
# Siapkan data
# -----------------------------------------
df = pd.read_csv(uploaded)
if "Date" not in df.columns or "Close" not in df.columns:
    st.error("CSV harus memiliki kolom Date dan Close.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

# -----------------------------------------
# Skala data dan buat window terakhir
# -----------------------------------------
all_scaled = scaler.transform(df[["Close"]].values)
if len(all_scaled) < lookback:
    st.error(f"Panjang data {len(all_scaled)} lebih kecil dari lookback {lookback}. Kurangi lookback atau tambahkan data.")
    st.stop()

last_window = all_scaled[-lookback:].reshape(1, lookback, 1).astype("float32")

# -----------------------------------------
# Prediksi ke depan recursive
# -----------------------------------------
preds_scaled = []
window = last_window.copy()
for _ in range(n_future):
    next_scaled = model.predict(window, verbose=0)[0, 0]
    preds_scaled.append(next_scaled)
    window = np.append(window[:, 1:, :], [[[next_scaled]]], axis=1)

preds_raw = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

# Buat tanggal ke depan
last_date = df["Date"].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=n_future + 1, freq=freq)[1:]

df_future = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": preds_raw
})

# -----------------------------------------
# Tampilkan hasil
# -----------------------------------------
st.subheader("Ringkasan")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah data historis", len(df))
c2.metric("Lookback", lookback)
c3.metric("Langkah ke depan", n_future)

st.subheader("Grafik historis dan forecast")
fig, ax = plt.subplots(figsize=(12, 5))
tail_df = df.tail(tail_n)
ax.plot(tail_df["Date"], tail_df["Close"], label="Historical Close")
ax.plot(df_future["Date"], df_future["Predicted_Close"], label="Forecasted Close", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Historis vs Forecast")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=True)

st.subheader("Data prediksi")
st.dataframe(df_future, use_container_width=True)

csv_bytes = df_future.to_csv(index=False).encode("utf-8")
st.download_button("Unduh CSV Prediksi", data=csv_bytes, file_name="prediksi_forward.csv", mime="text/csv")

st.success("Prediksi selesai.")