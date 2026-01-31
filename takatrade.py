import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras import backend as K 
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import gc 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- INITIALIZE CORE ENGINE ---
analyzer = SentimentIntensityAnalyzer()
pd.options.mode.chained_assignment = None

# --- 1. CONFIG & UI PREMIUM (Identik Code Lama) ---
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="logo_takatrade.png")

# Refresh otomatis 30 menit
st.markdown('<meta http-equiv="refresh" content="1800">', unsafe_allow_html=True)

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
    .news-card { background: #0a0a0a; border-left: 3px solid #FFD700; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
    .decision-box { text-align:center; padding:20px; border-radius:15px; background: rgba(10,10,10,0.8); margin-bottom:20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET LENGKAP ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "MATIC-USD", "LINK-USD", "PEPE-USD", "WIF-USD", "FLOKI-USD", "NEAR-USD", "ARB-USD"])
global_indices_forex = sorted(["EURUSD=X", "USDIDR=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F", "^JKSE", "^GSPC", "^IXIC"])
stock_us = sorted(["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"])
stock_id = sorted(["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK"])

database_aset = {"🌍 GLOBAL MARKET & FOREX": global_indices_forex, "💎 CRYPTOCURRENCY": crypto_list, "🇺🇸 US STOCKS": stock_us, "🇮🇩 INDONESIA STOCKS": stock_id}

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 25px; opacity: 0.85; font-weight: bold;'>ULTIMATE HYBRID SYSTEM</p>", unsafe_allow_html=True)
    selected = option_menu(None, ["Intelligence", "Radar", "Settings"], 
        icons=['cpu-fill', 'broadcast', 'gear-fill'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}})
    
    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    # Fitur 5 Menit - 1 Bulan Aktif
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    interval_map = {"5m":"5m", "15m":"15m", "30m":"30m", "1h":"60m", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}
    period_map = {"5m":"1d", "15m":"5d", "30m":"5d", "1h":"1mo", "1d":"2y", "1wk":"max", "1mo":"max"}
    
    if "epochs" not in st.session_state: st.session_state.epochs = 20
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- 4. ADVANCED HYBRID ENGINES ---

def get_multi_timeframe_confluence(ticker):
    """Mengecek tren di timeframe lebih besar (Code Lama)"""
    try:
        h1 = yf.download(ticker, period='5d', interval='60m', progress=False)
        d1 = yf.download(ticker, period='1mo', interval='1d', progress=False)
        if h1.empty or d1.empty: return "NEUTRAL"
        h1_trend = "UP" if h1['Close'].iloc[-1] > h1['Close'].rolling(20).mean().iloc[-1] else "DOWN"
        d1_trend = "UP" if d1['Close'].iloc[-1] > d1['Close'].rolling(20).mean().iloc[-1] else "DOWN"
        if h1_trend == "UP" and d1_trend == "UP": return "BULLISH CONFLUENCE 💎"
        if h1_trend == "DOWN" and d1_trend == "DOWN": return "BEARISH CONFLUENCE 📉"
        return "MIXED TREND ⚖️"
    except: return "NEUTRAL"

def add_pro_indicators(df):
    """Indikator Teknikal (Gabungan Code Lama & Baru)"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    # EMA & ATR
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    # Institutional Bridge (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df.ffill()

def train_hybrid_ai(ticker, interval, period, epochs):
    """Bi-LSTM + Sentiment Engine"""
    K.clear_session(); gc.collect()
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or len(df) < 40: return None, None, 0, []
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df = add_pro_indicators(df)
    
    # AI Training Logic
    features = ['Close', 'Volume', 'RSI']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    
    window = 30
    x, y = [], []
    for i in range(window, len(scaled_data)):
        x.append(scaled_data[i-window:i]); y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    
    model = Sequential([
        Input(shape=(window, len(features))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    
    # Forecast & Sentiment
    last_batch = scaled_data[-window:].reshape(1, window, len(features))
    pred_scaled = model.predict(last_batch, verbose=0)
    res_pred = scaler.inverse_transform([[pred_scaled[0,0], 0, 0]])[0,0]
    
    t_news = yf.Ticker(ticker)
    news = t_news.news[:3]
    sent_score = np.mean([analyzer.polarity_scores(n['title'])['compound'] for n in news]) if news else 0.0
    
    return df, res_pred, sent_score, news

# --- 5. MAIN DASHBOARD ---
if selected == "Intelligence":
    waktu_wib = datetime.now()
    st.markdown(f"### TAKATRADE Intelligence Pro | {waktu_wib.strftime('%H:%M:%S')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1: kat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2: pilihan = st.multiselect("🔎 Focus Assets", database_aset[kat], default=database_aset[kat][0])

    if st.button("RUN HYBRID ANALYSIS"):
        tabs = st.tabs(pilihan)
        for i, t in enumerate(pilihan):
            with tabs[i]:
                with st.spinner(f'Mengaktifkan Neural Network untuk {t}...'):
                    df, pred, sent, news = train_hybrid_ai(t, interval_map[horizon_label], period_map[horizon_label], st.session_state.epochs)
                    
                    if df is None:
                        st.error("Data tidak mencukupi untuk analisis ini.")
                        continue
                    
                    # Logika Metrics & Confluence
                    curr = df['Close'].iloc[-1]
                    pct = ((pred - curr) / curr) * 100
                    conf = get_multi_timeframe_confluence(t)
                    inst_flow = "ACCUMULATION 🏦" if df['OBV'].iloc[-1] > df['OBV'].rolling(10).mean().iloc[-1] else "DISTRIBUTION 🏛️"
                    
                    # Risk Management (Dari Code Lama)
                    risk_factor = 0.02 # 2%
                    atr = df['ATR'].iloc[-1]
                    sl_price = curr - (atr * 1.5) if pct > 0 else curr + (atr * 1.5)
                    tp_price = curr + (abs(curr - sl_price) * 2.5)
                    qty = (st.session_state.modal * risk_factor) / abs(curr - sl_price)

                    # --- UI Metrics ---
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Live Price", f"{curr:,.2f}")
                    m2.metric(f"AI Target ({horizon_label})", f"{pred:,.2f}", f"{pct:+.2f}%")
                    m3.metric("Sentimen", f"{sent:+.2f}")
                    m4.metric("Inst. Flow", "ACCUM" if "ACCUM" in inst_flow else "DISTRIB")

                    # --- Decision Matrix (Hybrid) ---
                    score = 0
                    if pct > 0.4: score += 1
                    if "BULLISH" in conf: score += 1
                    if sent > 0.05: score += 1
                    if df['RSI'].iloc[-1] < 65: score += 0.5
                    
                    if score >= 3: action, color = "STRONG BUY 🚀", "#00FFCC"
                    elif score >= 2: action, color = "BUY 🟢", "#00FFCC"
                    elif score <= 1: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "NEUTRAL ⚖️", "#FFA500"

                    st.markdown(f"""
                        <div class="decision-box" style="border: 1px solid {color};">
                            <h2 style="color:{color}; margin:0;">{action}</h2>
                            <p style="color:#FFD700; margin:5px 0;">Optimized Qty: {qty:.4f} Units | Risk: 2%</p>
                            <small style="color:gray;">TP: {tp_price:,.2f} | SL: {sl_price:,.2f} | Confluence: {conf}</small>
                        </div>
                    """, unsafe_allow_html=True)

                    # --- Charting ---
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index[-60:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
                    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['EMA9'], line=dict(color='#FFD700', width=1), name="EMA 9"))
                    fig.update_layout(template="plotly_dark", height=450, paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # News Feed
                    if news:
                        with st.expander("📰 Market Intelligence Feed"):
                            for n in news:
                                st.markdown(f"<div class='news-card'><b>{n['title']}</b><br><small>{n['publisher']}</small></div>", unsafe_allow_html=True)

elif selected == "Radar":
    st.markdown(f"### 📡 Deep Scanning Radar ({horizon_label})")
    radar_kat = st.selectbox("Pilih Universe untuk Di-scan", list(database_aset.keys()))
    if st.button("MULAI SCANNING"):
        results = []
        progress = st.progress(0)
        assets = database_aset[radar_kat]
        for idx, ticker in enumerate(assets):
            progress.progress((idx + 1) / len(assets))
            df_r, pred_r, sent_r, _ = train_hybrid_ai(ticker, interval_map[horizon_label], period_map[horizon_label], 5)
            if df_r is not None:
                curr_r = df_r['Close'].iloc[-1]
                pct_r = ((pred_r - curr_r) / curr_r) * 100
                conf_r = get_multi_timeframe_confluence(ticker)
                results.append({"Asset": ticker, "Price": round(curr_r, 2), "Forecast": f"{pct_r:+.2f}%", "Confluence": conf_r, "Signal": "BUY" if pct_r > 0.5 else "SELL"})
        st.table(pd.DataFrame(results))

else:
    st.title("⚙️ System Settings")
    st.session_state.epochs = st.slider("Neural Network Training Epochs", 10, 100, 20)
    st.session_state.modal = st.number_input("Investment Modal ($)", value=1000)

st.caption("TAKATRADE PRO ULTIMATE © 2026 | Bidirectional LSTM & Sentiment Engine")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.







































