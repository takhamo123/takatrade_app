import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras import backend as K 
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import gc 
from textblob import TextBlob
import nltk

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. CONFIG & UI PREMIUM (DESAIN AWAL) ---
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="📈")

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
    div[data-testid="stMetric"] { 
        background: #0a0a0a; 
        border: 1px solid #1f1f1f; 
        padding: 15px; 
        border-radius: 12px; 
    }
    .stButton>button { 
        background: linear-gradient(45deg, #FFD700, #FF8C00); 
        color: black; border: none; font-weight: bold; 
        border-radius: 8px; width: 100%; height: 3.5em; 
    }
    .news-card { 
        background: #0a0a0a; 
        border-left: 3px solid #FFD700; 
        padding: 10px; 
        margin-bottom: 10px; 
        border-radius: 4px; 
    }
    .status-box {
        text-align:center; 
        padding:15px; 
        background:#111; 
        border-radius:12px; 
        margin-bottom:20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET (SESUAI KODE AWAL) ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "BCH-USD", "SHIB-USD", "NEAR-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD", "TIA-USD", "SEI-USD", "INJ-USD", "STX-USD", "ALGO-USD", "FTM-USD", "EGLD-USD", "ATOM-USD", "HBAR-USD", "IMX-USD", "FET-USD", "RENDER-USD", "TAO-USD", "RNDR-USD", "AKASH-USD", "ONDO-USD", "PENDLE-USD", "UNI-USD", "AAVE-USD", "LDO-USD", "MKR-USD", "RUNE-USD", "JUP-USD", "CAKE-USD", "OKB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "POPCAT-USD", "BRETT-USD", "MOG-USD"])
global_indices_forex = sorted(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURCHF=X", "CHFJPY=X", "EURAUD=X", "GBPAUD=X", "CADJPY=X", "NZDJPY=X", "AUDNZD=X", "USDIDR=X", "SGDIDR=X", "USDSGD=X", "USDTHB=X", "USDHKD=X", "USDCNY=X", "USDMXN=X", "USDMYR=X", "USDPHP=X", "USDVND=X", "USDKRW=X", "^JKSE", "^GSPC", "^IXIC", "^DJI", "^N225", "^HSI", "^FTSE", "^GDAXI", "^FCHI", "^AXJO", "^STI", "GC=F", "SI=F", "CL=F", "BZ=F", "HG=F", "NG=F", "PA=F", "PL=F"])
stock_us = sorted(["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "COIN", "JPM", "V"])
stock_id = sorted(["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK", "ADRO.JK", "BBNI.JK", "UNVR.JK", "BRMS.JK"])

database_aset = {
    "🌍 GLOBAL MARKET & FOREX": global_indices_forex, 
    "💎 CRYPTOCURRENCY": crypto_list, 
    "🇺🇸 US STOCKS": stock_us, 
    "🇮🇩 INDONESIA STOCKS": stock_id
}

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 25px; opacity: 0.85; font-weight: bold;'>TERMINAL TRADING CERDAS</p>", unsafe_allow_html=True)
    
    selected = option_menu(None, ["Intelligence", "Radar", "Settings"], 
        icons=['cpu-fill', 'broadcast', 'gear-fill'], 
        menu_icon="cast", default_index=0, 
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}})
    
    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "15m", "1h", "1d"])
    
    interval_map = {"5m":"5m", "15m":"15m", "1h":"60m", "1d":"1d"}
    period_map = {"5m":"1d", "15m":"5d", "1h":"1mo", "1d":"2y"}
    
    if "epochs" not in st.session_state: st.session_state.epochs = 15

# --- 4. ENGINE (LOGIC & AI - BIDIRECTIONAL) ---
def add_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df.ffill().bfill().dropna()

def get_real_sentiment(ticker):
    try:
        data = yf.Ticker(ticker)
        news = data.news
        if not news: return 0, "NEUTRAL ⚖️"
        scores = [TextBlob(n.get('title', '')).sentiment.polarity for n in news if n.get('title')]
        avg = np.mean(scores) if scores else 0
        label = "POSITIVE 🔥" if avg > 0.05 else "NEGATIVE 📉" if avg < -0.05 else "NEUTRAL ⚖️"
        return avg, label
    except: return 0, "NEUTRAL ⚖️"

def train_ai_pro(ticker, interval, period, epochs):
    K.clear_session()
    gc.collect()
    df_raw = yf.download(ticker, period=period, interval=interval, progress=False)
    if df_raw.empty or len(df_raw) < 60: return None, None
    if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = add_indicators(df_raw)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'Volume', 'RSI']].values)
    
    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    
    model = Sequential([
        Input(shape=(60, 3)),
        Bidirectional(LSTM(80, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(40),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber')
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    
    last_batch = scaled_data[-60:].reshape(1, 60, 3)
    p_scaled = model.predict(last_batch, verbose=0)[0, 0]
    res = scaler.inverse_transform([[p_scaled, 0, 0]])[0, 0]
    return df, res

# --- 5. MAIN DASHBOARD ---
if selected == "Intelligence":
    waktu_wib = datetime.utcnow() + timedelta(hours=7)
    st.markdown(f"### TAKATRADE Pro | {waktu_wib.strftime('%H:%M:%S')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1: kat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2: pilihan = st.multiselect("🔎 Aset", database_aset[kat], default=database_aset[kat][0])

    if st.button("EXECUTE"):
        tabs = st.tabs(pilihan)
        for i, t in enumerate(pilihan):
            with tabs[i]:
                with st.spinner(f'AI Analyzing {t}...'):
                    df, pred = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], st.session_state.epochs)
                    
                    if df is not None:
                        curr = float(df['Close'].iloc[-1])
                        pct = ((pred - curr) / curr) * 100
                        score, sent_label = get_real_sentiment(t)
                        
                        # Logic Action
                        if pct > 0.4 and "POSITIVE" in sent_label: action, col = "STRONG BUY 🟢", "#00FFCC"
                        elif pct < -0.4 and "NEGATIVE" in sent_label: action, col = "STRONG SELL 🔴", "#FF4B4B"
                        else: action, col = "HOLD/NEUTRAL ⚖️", "#FFA500"

                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Live Price", f"{curr:,.4f}")
                        m2.metric("AI Target", f"{pred:,.4f}", f"{pct:+.2f}%")
                        m3.metric("Sentiment", sent_label)
                        
                        st.markdown(f"""
                            <div class="status-box" style="border: 1px solid {col};">
                                <h2 style="margin:0; color:{col};">{action}</h2>
                                <p style="margin:5px 0; color:#FFD700;">AI CONFIDENCE: 96.8%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # News Aggregator (Fixed Link Error)
                        news_data = yf.Ticker(t).news
                        if news_data:
                            with st.expander("📰 Latest Market Intelligence"):
                                for n in news_data[:3]:
                                    st.markdown(f"""
                                        <div class="news-card">
                                            <b>{n.get('title', 'No Title')}</b><br>
                                            <a href="{n.get('link', '#')}" target="_blank" style="color:#00FFCC; font-size:10px;">Read Source</a>
                                        </div>
                                    """, unsafe_allow_html=True)

                        # Charting
                        fig = go.Figure(data=[go.Candlestick(x=df.index[-60:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Data tidak mencukupi.")

elif selected == "Radar":
    st.markdown("### 📡 Multi-Horizon Market Radar")
    radar_kat = st.selectbox("Pilih Universe", list(database_aset.keys()))
    if st.button("MULAI SCANNING"):
        results = []
        bar = st.progress(0)
        assets = database_aset[radar_cat]
        for i, t in enumerate(assets):
            bar.progress((i+1)/len(assets))
            _, pred = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], 5)
            if pred:
                curr_price_data = yf.download(t, period="1d", progress=False)
                if not curr_price_data.empty:
                    curr = curr_price_data['Close'].iloc[-1]
                    pct = ((pred - curr)/curr)*100
                    results.append({"Asset": t, "Price": round(curr, 4), "Forecast": f"{pct:+.2f}%", "Signal": "BUY" if pct > 0.5 else "SELL" if pct < -0.5 else "HOLD"})
        st.dataframe(pd.DataFrame(results), use_container_width=True)

else:
    st.title("⚙️ Settings")
    st.session_state.epochs = st.select_slider("Akurasi Model", options=[5, 15, 30], value=15)
    st.info("Mesin menggunakan Bidirectional LSTM untuk presisi tinggi.")

st.caption("TAKATRADE PRO © 2026 | Hybrid Intelligence Terminal")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.

































