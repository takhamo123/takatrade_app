import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta

# --- 1. CONFIG & UI PREMIUM (TETAP ASLI) ---
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="logo_takatrade.png")

st.markdown("""
    <style>
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .stApp { background-color: #000000; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1a1a1a; }
    .logo-container {
        font-family: 'Syncopate', sans-serif;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px; font-weight: bold; text-align: center;
        padding: 15px 0; letter-spacing: 5px;
    }
    div[data-testid="stMetric"] { background: #0a0a0a; border: 1px solid #1f1f1f; padding: 15px; border-radius: 12px; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #FF8C00); color: black; border: none; font-weight: bold; border-radius: 8px; width: 100%; height: 3.5em; }
    
    @media (max-width: 640px) {
        .logo-container { font-size: 20px; letter-spacing: 2px; }
        div[data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET (DITAMBAHKAN FOREX LENGKAP) ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD", "LINK-USD", "NEAR-USD", "ARB-USD", "SUI-USD", "PEPE-USD", "RENDER-USD", "FET-USD"])
stock_us = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"]
stock_id = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK", "ADRO.JK"]
# Penambahan Aset Forex Lengkap
forex_commo = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "USDIDR=X", "SGDIDR=X", "EURJPY=X", "GBPJPY=X", "GC=F", "CL=F"
]

database_aset = {
    "💎 CRYPTOCURRENCY": crypto_list,
    "🇺🇸 US STOCKS": stock_us,
    "🇮🇩 INDONESIA STOCKS": stock_id,
    "🌍 FOREX & COMMO": sorted(forex_commo)
}

# --- 3. SIDEBAR NAVIGATION (TETAP ASLI) ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 20px; opacity: 0.8;'>
        TERMINAL TRADING CERDAS
        </p>
    """, unsafe_allow_html=True)

    selected = option_menu(None, ["Intelligence", "Settings"], 
        icons=['cpu-fill', 'gear-fill'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}})

    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["1d", "7d", "30d"])
    steps = 1 if horizon_label == "1d" else 7 if horizon_label == "7d" else 30
    
    if "epochs" not in st.session_state: st.session_state.epochs = 12
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- 4. ENGINE AI (TETAP ASLI) ---
@st.cache_resource(show_spinner=False)
def train_ai_pro(ticker, steps, epochs):
    df = yf.download(ticker, period='3y', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.ffill()
    
    scaler = MinMaxScaler()
    # Penanganan volume untuk forex
    if 'Volume' not in df.columns: df['Volume'] = 0
    scaled_data = scaler.fit_transform(df[['Close', 'Volume']].values)
    
    x, y, window = [], [], 60
    for i in range(window, len(scaled_data)):
        x.append(scaled_data[i-window:i])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    
    model = Sequential([
        Input(shape=(window, 2)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    
    test_batch = scaled_data[-(window+30):-30]
    bt_preds = []
    for _ in range(30):
        p = model.predict(test_batch.reshape(1, window, 2), verbose=0)
        bt_preds.append(p[0, 0])
        new_entry = np.array([[p[0, 0], scaled_data[-30+len(bt_preds)-1, 1]]])
        test_batch = np.append(test_batch[1:], new_entry, axis=0)
    
    accuracy = 100 - (np.mean(np.abs(scaled_data[-30:, 0] - np.array(bt_preds))) * 100)
    
    last_batch = scaled_data[-window:].tolist()
    preds = []
    for _ in range(steps):
        p = model.predict(np.array(last_batch[-window:]).reshape(1, window, 2), verbose=0)
        preds.append(p[0, 0])
        last_batch.append([p[0, 0], last_batch[-1][1]])
        
    res_preds = scaler.inverse_transform(np.column_stack([preds, [0]*steps]))[:, 0]
    return df, res_preds, accuracy

# --- 5. MAIN DASHBOARD (TETAP ASLI + SENTIMEN) ---
if selected == "Intelligence":
    waktu_wib = datetime.utcnow() + timedelta(hours=7)
    st.markdown(f"### TAKATRADE Pro | {waktu_wib.strftime('%H:%M')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        kat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2:
        pilihan = st.multiselect("🔎 Aset", database_aset[kat], default=database_aset[kat][0])

    if st.button("🔥 JALANKAN ANALISA QUANT"):
        tabs = st.tabs(pilihan)
        for i, t in enumerate(pilihan):
            with tabs[i]:
                with st.spinner(f'AI memproses {t}...'):
                    hist, preds, acc = train_ai_pro(t, steps, st.session_state.epochs)
                    curr, target = float(hist['Close'].iloc[-1]), float(preds[-1])
                    pct = ((target - curr) / curr) * 100
                    
                    # Logika Sentimen Sesuai Kebutuhan
                    vol_chg = (hist['Volume'].iloc[-1] / hist['Volume'].rolling(10).mean().iloc[-1]) if hist['Volume'].iloc[-1] != 0 else 1
                    sentiment = "POSITIVE ✨" if pct > 0 and vol_chg > 1 else "NEGATIVE ⚠️" if pct < 0 else "NEUTRAL ⚖️"

                    if pct > 1.5 and vol_chg > 1: action, color = "STRONG BUY 🟢", "#00FFCC"
                    elif pct > 0: action, color = "BUY 🟢", "#00FFCC"
                    elif pct < -1.5: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "SELL 🔴", "#FF4B4B"

                    m1, m2 = st.columns(2)
                    m1.metric("Price", f"{curr:,.2f}")
                    m2.metric(f"Target ({horizon_label})", f"{target:,.2f}", f"{pct:+.2f}%")
                    m3, m4 = st.columns(2)
                    m3.metric("AI Confidence", f"{acc:.1f}%")
                    m4.metric("Sentiment", sentiment) # SENTIMEN MUNCUL KEMBALI DI SINI
                    
                    st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border:1px solid {color}; border-radius:10px; margin-bottom:20px;'><h2 style='margin:0; color:{color};'>{action}</h2></div>", unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index[-60:], open=hist['Open'].iloc[-60:], high=hist['High'].iloc[-60:], low=hist['Low'].iloc[-60:], close=hist['Close'].iloc[-60:], name="Market"))
                    f_dates = [hist.index[-1] + timedelta(days=x) for x in range(1, steps + 1)]
                    fig.add_trace(go.Scatter(x=f_dates, y=preds, name="AI Path", line=dict(color='#FFD700', width=3, dash='dot')))
                    
                    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=5, r=5, t=30, b=5), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"💡 **AI Logic:** Akurasi backtest {acc:.1f}%. Analisa Volume menunjukkan tren {'kuat' if vol_chg > 1 else 'lemah'}.")

else:
    st.title("⚙️ Settings")
    ai_speed = st.select_slider("Akurasi Model", options=["Fast", "Balanced", "Precision"], value="Balanced")
    st.session_state.epochs = 5 if ai_speed == "Fast" else 15 if ai_speed == "Balanced" else 30
    st.session_state.modal = st.number_input("Modal Investasi ($)", value=1000)

st.caption("TAKATRADE PRO © 2026 | Terminal Trading Cerdas Berbasis Deep Learning")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.









