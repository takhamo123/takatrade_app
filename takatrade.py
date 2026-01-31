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

# Menghilangkan warning dekoratif pandas
pd.options.mode.chained_assignment = None

# --- 1. CONFIG & UI PREMIUM (Identik Code Awal) ---
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
    @media (max-width: 640px) {
        .logo-container { font-size: 20px; letter-spacing: 2px; }
        div[data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET (100% SESUAI CODE AWAL) ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "BCH-USD", "SHIB-USD", "NEAR-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD", "TIA-USD", "SEI-USD", "INJ-USD", "STX-USD", "ALGO-USD", "FTM-USD", "EGLD-USD", "ATOM-USD", "HBAR-USD", "IMX-USD", "FET-USD", "RENDER-USD", "TAO-USD", "RNDR-USD", "AKASH-USD", "ONDO-USD", "PENDLE-USD", "UNI-USD", "AAVE-USD", "LDO-USD", "MKR-USD", "RUNE-USD", "JUP-USD", "CAKE-USD", "OKB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "POPCAT-USD", "BRETT-USD", "MOG-USD"])
global_indices_forex = sorted(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURCHF=X", "CHFJPY=X", "EURAUD=X", "GBPAUD=X", "CADJPY=X", "NZDJPY=X", "AUDNZD=X", "USDIDR=X", "SGDIDR=X", "USDSGD=X", "USDTHB=X", "USDHKD=X", "USDCNY=X", "USDMXN=X", "USDMYR=X", "USDPHP=X", "USDVND=X", "USDKRW=X", "^JKSE", "^GSPC", "^IXIC", "^DJI", "^N225", "^HSI", "^FTSE", "^GDAXI", "^FCHI", "^AXJO", "^STI", "GC=F", "SI=F", "CL=F", "BZ=F", "HG=F", "NG=F", "PA=F", "PL=F"])
stock_us = sorted(["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "COIN", "JPM", "V"])
stock_id = sorted(["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK", "ADRO.JK", "BBNI.JK", "UNVR.JK", "BRMS.JK"])

database_aset = {"🌍 GLOBAL MARKET & FOREX": global_indices_forex, "💎 CRYPTOCURRENCY": crypto_list, "🇺🇸 US STOCKS": stock_us, "🇮🇩 INDONESIA STOCKS": stock_id}

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
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "10m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    interval_map = {"5m":"5m", "10m":"2m", "15m":"15m", "30m":"30m", "1h":"60m", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}
    period_map = {"5m":"1d", "10m":"1d", "15m":"5d", "30m":"5d", "1h":"1mo", "1d":"3y", "1wk":"max", "1mo":"max"}
    if "epochs" not in st.session_state: st.session_state.epochs = 12
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- HELPER: INDICATOR ENGINE ---
def add_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower'] = df['MA20'] - (df['STD20'] * 2)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

# --- 4. ENGINE AI (BIDIRECTIONAL LSTM - MEMORY OPTIMIZED) ---
def train_ai_pro(ticker, interval, period, steps, epochs):
    K.clear_session()
    gc.collect()

    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or len(df) < 40: return None, None, 0
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.ffill()
    if 'Volume' not in df.columns: df['Volume'] = 0
    
    df = add_indicators(df)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'Volume']].values)
    
    window = 30 # Optimized window
    x, y = [], []
    for i in range(window, len(scaled_data)):
        x.append(scaled_data[i-window:i]); y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    
    # Brain: Bidirectional LSTM (Peningkatan dari Code Awal)
    model = Sequential([
        Input(shape=(window, 2)), 
        Bidirectional(LSTM(64, return_sequences=True)), 
        Dropout(0.2), 
        LSTM(32), 
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    
    # Backtest Accuracy Calculation
    test_len = 10 if len(scaled_data) > 10 else 1
    test_batch = scaled_data[-window-test_len:-test_len]
    bt_preds = []
    for _ in range(test_len):
        p = model.predict(test_batch.reshape(1, window, 2), verbose=0)
        bt_preds.append(p[0, 0])
        new_entry = np.array([[p[0, 0], scaled_data[-test_len+len(bt_preds)-1, 1]]])
        test_batch = np.append(test_batch[1:], new_entry, axis=0)
    accuracy = 100 - (np.mean(np.abs(scaled_data[-test_len:, 0] - np.array(bt_preds))) * 100)
    
    # Forecast Future
    last_batch = scaled_data[-window:].tolist()
    preds = []
    for _ in range(steps):
        p = model.predict(np.array(last_batch[-window:]).reshape(1, window, 2), verbose=0)
        preds.append(p[0, 0])
        last_batch.append([p[0, 0], last_batch[-1][1]])
    res_preds = scaler.inverse_transform(np.column_stack([preds, [0]*steps]))[:, 0]

    return df, res_preds, accuracy

# --- 5. ADVANCED MODULES ---
def get_multi_timeframe_confluence(ticker):
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

def get_news_aggregator(ticker):
    try:
        data = yf.Ticker(ticker)
        return data.news[:3]
    except: return []

# --- 6. MAIN DASHBOARD ---
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
                with st.spinner(f'AI memproses {t}...'):
                    df_raw, preds, acc = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], 1, st.session_state.epochs)
                    if df_raw is None: continue
                    
                    confluence = get_multi_timeframe_confluence(t)
                    news = get_news_aggregator(t)
                    
                    curr, target = float(df_raw['Close'].iloc[-1]), float(preds[-1])
                    pct = ((target - curr) / curr) * 100
                    rsi_now = df_raw['RSI'].iloc[-1]
                    
                    # Sentiment Hybrid Logic
                    vol_avg = df_raw['Volume'].rolling(10).mean().iloc[-1]
                    vol_chg = (df_raw['Volume'].iloc[-1] / vol_avg) if vol_avg != 0 else 1
                    if rsi_now > 70: sentiment = "OVERBOUGHT ⚠️"
                    elif rsi_now < 30: sentiment = "OVERSOLD ✨"
                    else: sentiment = "BULLISH VOL 🔥" if vol_chg > 1.2 and pct > 0 else "NEUTRAL ⚖️"

                    # Action Logic with Confluence
                    if pct > 1.0 and rsi_now < 65 and "BULLISH" in confluence: action, color = "STRONG BUY 🟢", "#00FFCC"
                    elif pct > 0: action, color = "BUY 🟢", "#00FFCC"
                    elif pct < -1.0: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "HOLD/NEUTRAL ⚖️", "#FFA500"
                    
                    # Risk Management (Code Awal)
                    risk_factor = 0.02
                    atr_proxy = df_raw['Close'].rolling(14).std().iloc[-1] * 2
                    sl_price = curr - atr_proxy if action.startswith("BUY") else curr + atr_proxy
                    tp_price = curr + (abs(curr - sl_price) * 2.5)
                    inst_flow = "ACCUMULATION 🏦" if df_raw['OBV'].iloc[-1] > df_raw['OBV'].rolling(10).mean().iloc[-1] else "DISTRIBUTION 🏛️"

                    # UI Metrics
                    m1, m2 = st.columns(2); m1.metric("Live Price", f"{curr:,.4f}"); m2.metric(f"AI Target ({horizon_label})", f"{target:,.4f}", f"{pct:+.2f}%")
                    m3, m4 = st.columns(2); m3.metric("MTF Confluence", confluence); m4.metric("Inst. Flow", inst_flow)
                    
                    st.markdown(f"""
                        <div style='text-align:center; padding:15px; background:#111; border:1px solid {color}; border-radius:12px; margin-bottom:20px;'>
                            <h2 style='margin:0; color:{color};'>{action}</h2>
                            <p style='margin:5px 0; color:#FFD700; font-weight:bold;'>OPTIMIZED QTY: ${(st.session_state.modal * risk_factor / abs(curr - sl_price if curr!=sl_price else 1)):.2f} Units</p>
                            <p style='margin:0; color:gray; font-size:12px;'>DYN-TP: {tp_price:,.4f} | DYN-SL: {sl_price:,.4f} | RR: 1:2.5</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # News Aggregator
                    if news:
                        with st.expander("📰 Latest Market Intelligence"):
                            for n in news:
                                st.markdown(f"<div class='news-card'><b>{n['title']}</b><br><small>{datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d %H:%M')}</small></div>", unsafe_allow_html=True)

                    # Plotting
                    fig = go.Figure(data=[go.Candlestick(x=df_raw.index[-60:], open=df_raw['Open'], high=df_raw['High'], low=df_raw['Low'], close=df_raw['Close'], name="Market")])
                    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, paper_bgcolor='black', plot_bgcolor='black')
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"💡 **AI Logic:** Confluence {confluence} detected. Sentiment: {sentiment}. Akurasi Model: {acc:.2f}%")

elif selected == "Radar":
    st.markdown("### 📡 Multi-Horizon Market Radar")
    radar_kat = st.selectbox("Universe", list(database_aset.keys()))
    if st.button("MULAI SCANNING"):
        results = []
        progress = st.progress(0)
        assets = database_aset[radar_kat]
        for idx, ticker in enumerate(assets):
            progress.progress((idx + 1) / len(assets))
            df_r, preds_r, _ = train_ai_pro(ticker, interval_map[horizon_label], period_map[horizon_label], 1, 5)
            if df_r is not None:
                curr_r = df_raw['Close'].iloc[-1] if 'df_raw' in locals() else df_r['Close'].iloc[-1]
                pct_r = ((preds_r[-1] - curr_r) / curr_r) * 100
                results.append({"Aset": ticker, "Price": round(curr_r, 4), "Forecast %": f"{pct_r:+.2f}%", "Signal": "BUY" if pct_r > 0 else "SELL"})
        st.dataframe(pd.DataFrame(results), use_container_width=True)

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








































