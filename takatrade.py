import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K 
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import gc 
from textblob import TextBlob
import nltk

# Inisialisasi NLTK untuk Sentiment Analysis
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Konfigurasi Dasar
pd.options.mode.chained_assignment = None
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="📈")

# --- 1. STYLING UI ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; }
    .logo-container {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; font-weight: bold; text-align: center;
        padding: 10px 0; letter-spacing: 3px;
    }
    .news-card { background: #0a0a0a; border-left: 3px solid #FFD700; padding: 12px; margin-bottom: 10px; border-radius: 8px; }
    div[data-testid="stMetric"] { background: #0d0d0d; border: 1px solid #222; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET ---
database_aset = {
    "🌍 GLOBAL & FOREX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "^JKSE", "^GSPC"],
    "💎 CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"],
    "🇺🇸 US STOCKS": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META"],
    "🇮🇩 INDO STOCKS": ["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "GOTO.JK"]
}

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    selected = option_menu(None, ["Intelligence", "Radar", "Settings"], 
        icons=['cpu', 'broadcast', 'gear'], default_index=0,
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black"}})
    
    st.markdown("---")
    horizon = st.select_slider("Time Horizon", options=["5m", "15m", "1h", "1d"], value="1h")
    
    # Mapping Period & Interval
    mapping = {"5m": ("1d", "5m"), "15m": ("5d", "15m"), "1h": ("1mo", "60m"), "1d": ("2y", "1d")}
    period_val, interval_val = mapping[horizon]

# --- 4. CORE ENGINE (AI & LOGIC) ---
def get_indicators(df):
    df = df.copy()
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    # Moving Average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df.ffill().bfill().dropna()

def train_predict(ticker, p, i, ep):
    K.clear_session()
    gc.collect() # Force garbage collection to save memory
    
    df = yf.download(ticker, period=p, interval=i, progress=False)
    if df.empty or len(df) < 60: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df_ind = get_indicators(df)
    scaler = RobustScaler()
    # Menggunakan fitur Close, Volume, dan RSI untuk input AI
    data_scaled = scaler.fit_transform(df_ind[['Close', 'Volume', 'RSI']].values)
    
    x, y = [], []
    for j in range(60, len(data_scaled)):
        x.append(data_scaled[j-60:j])
        y.append(data_scaled[j, 0])
    
    x, y = np.array(x), np.array(y)

    model = Sequential([
        Input(shape=(60, 3)),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='huber')
    model.fit(x, y, epochs=ep, batch_size=32, verbose=0)
    
    last_win = data_scaled[-60:].reshape(1, 60, 3)
    pred_val = model.predict(last_win, verbose=0)[0, 0]
    
    # Reversing scaling hanya untuk kolom 'Close'
    final_pred = scaler.inverse_transform([[pred_val, 0, 0]])[0, 0]
    
    return df_ind, final_pred

def fetch_sentiment(ticker):
    try:
        t_data = yf.Ticker(ticker)
        raw_news = t_data.news
        if not raw_news: return 0, "NEUTRAL ⚖️"
        scores = [TextBlob(n.get('title', '')).sentiment.polarity for n in raw_news if n.get('title')]
        avg = np.mean(scores) if scores else 0
        lbl = "POSITIVE 🔥" if avg > 0.05 else "NEGATIVE 📉" if avg < -0.05 else "NEUTRAL ⚖️"
        return avg, lbl
    except: return 0, "NEUTRAL ⚖️"

# --- 5. MAIN INTERFACE ---
if selected == "Intelligence":
    st.subheader(f"🧠 AI Market Intelligence - {horizon}")
    cat = st.selectbox("Pilih Sektor", list(database_aset.keys()))
    assets = st.multiselect("Pilih Aset", database_aset[cat], default=database_aset[cat][0] if database_aset[cat] else None)

    if st.button("RUN ANALYSIS"):
        for t in assets:
            with st.status(f"Menganalisis {t}...", expanded=True) as status:
                ep_val = st.session_state.get('epochs', 12)
                df, pred = train_predict(t, period_val, interval_val, ep_val)
                
                if df is not None:
                    curr = float(df['Close'].iloc[-1])
                    diff = ((pred - curr) / curr) * 100
                    score, label = fetch_sentiment(t)
                    
                    # Logic Signal
                    if diff > 0.5 and label == "POSITIVE 🔥": 
                        sig, col = "STRONG BUY 🟢", "#00FFCC"
                    elif diff < -0.5 and label == "NEGATIVE 📉": 
                        sig, col = "STRONG SELL 🔴", "#FF4B4B"
                    elif diff > 0:
                        sig, col = "NEUTRAL BUY ⚖️", "#AAFF00"
                    else: 
                        sig, col = "NEUTRAL SELL ⚖️", "#FFA500"

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Live Price", f"{curr:,.2f}")
                    col2.metric("AI Target", f"{pred:,.2f}", f"{diff:+.2f}%")
                    col3.metric("Sentiment", label)

                    st.markdown(f"<div style='border: 2px solid {col}; padding:15px; border-radius:10px; text-align:center; background: rgba(0,0,0,0.5);'><h2 style='color:{col};'>{sig}</h2></div>", unsafe_allow_html=True)
                    
                    # Candlestick
                    df_plot = df.tail(50)
                    fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Market")])
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Gagal memuat data atau data {t} terlalu sedikit.")

elif selected == "Radar":
    st.subheader("📡 Radar Signal Scanner")
    radar_cat = st.selectbox("Scan Universe", list(database_aset.keys()))
    if st.button("START SCAN"):
        results = []
        bar = st.progress(0)
        assets_to_scan = database_aset[radar_cat]
        for i, t in enumerate(assets_to_scan):
            bar.progress((i+1)/len(assets_to_scan))
            # Fast train for radar
            _, pred = train_predict(t, period_val, interval_val, 5) 
            if pred:
                last_data = yf.download(t, period="1d", interval="1m", progress=False)
                if not last_data.empty:
                    curr = float(last_data['Close'].iloc[-1])
                    pct = ((pred - curr)/curr)*100
                    action = "BUY 🟢" if pct > 0.6 else "SELL 🔴" if pct < -0.6 else "HOLD ⚖️"
                    results.append({"Asset": t, "Price": round(curr, 4), "Forecast": f"{pct:+.2f}%", "Action": action})
        
        if results:
            st.table(pd.DataFrame(results))
        else:
            st.warning("Tidak ada data yang ditemukan saat scan.")

else:
    st.title("⚙️ Settings")
    st.session_state.epochs = st.slider("Model Epochs (Akurasi)", 5, 50, 15)
    st.info("Tips: Gunakan Epochs 15 untuk keseimbangan kecepatan dan akurasi. Gunakan 30+ untuk analisis mendalam.")
    st.markdown("---")
    st.caption("TAKATRADE PRO Engine v2.1 (Stability Patch)")

st.caption("© 2026 TAKATRADE PRO | Precise AI Terminal")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.
































