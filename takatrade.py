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

# --- 1. CONFIG & UI PREMIUM (100% Identik Code Awal) ---
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
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET (FULL SESUAI CODE AWAL) ---
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
        icons=['cpu-fill', 'broadcast', 'gear-fill'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}})
    
    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "10m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    interval_map = {"5m":"5m", "10m":"2m", "15m":"15m", "30m":"30m", "1h":"60m", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}
    period_map = {"5m":"1d", "10m":"1d", "15m":"5d", "30m":"5d", "1h":"1mo", "1d":"3y", "1wk":"max", "1mo":"max"}
    
    if "epochs" not in st.session_state: st.session_state.epochs = 12
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- 4. ENGINE INDIKATOR (Bollinger, MACD, OBV, RSI) ---
def add_advanced_indicators(df):
    # RSI & Bollinger Bands
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss)))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Institutional (OBV & ATR)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    return df

# --- 5. ENGINE AI (MEMORY OPTIMIZED Bi-LSTM) ---
def train_ai_pro(ticker, interval, period, epochs):
    K.clear_session(); gc.collect()
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty or len(df) < 40: return None, None, 0
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df = add_advanced_indicators(df).ffill()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close', 'Volume', 'RSI', 'MACD']].values)
    
    window = 30
    x, y = [], []
    for i in range(window, len(scaled)):
        x.append(scaled[i-window:i]); y.append(scaled[i, 0])
    
    model = Sequential([
        Input(shape=(window, 4)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(x), np.array(y), epochs=epochs, batch_size=32, verbose=0)
    
    # Forecast
    pred = model.predict(scaled[-window:].reshape(1, window, 4), verbose=0)
    res_pred = scaler.inverse_transform([[pred[0,0], 0, 0, 0]])[0,0]
    
    return df, res_pred, 95.0 # Return accuracy dummy atau hitung manual

# --- 6. MAIN DASHBOARD ---
if selected == "Intelligence":
    waktu_wib = datetime.utcnow() + timedelta(hours=7)
    st.markdown(f"### TAKATRADE Pro | {waktu_wib.strftime('%H:%M:%S')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1: kat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2: pilihan = st.multiselect("🔎 Aset", database_aset[kat], default=database_aset[kat][0])

    if st.button("EXECUTE ANALYSIS"):
        tabs = st.tabs(pilihan)
        for i, t in enumerate(pilihan):
            with tabs[i]:
                with st.spinner(f'AI memproses {t}...'):
                    df, pred, acc = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], st.session_state.epochs)
                    if df is None: continue
                    
                    curr = df['Close'].iloc[-1]
                    pct = ((pred - curr) / curr) * 100
                    
                    # Logika Strategi (MACD & Bollinger)
                    macd_bull = df['MACD'].iloc[-1] > df['Signal'].iloc[-1]
                    bb_low = curr < (df['Lower'].iloc[-1] * 1.01)
                    inst_flow = "ACCUMULATION 🏦" if df['OBV'].iloc[-1] > df['OBV'].rolling(10).mean().iloc[-1] else "DISTRIBUTION 🏛️"

                    # Action & Color Logic
                    if pct > 0.8 and macd_bull: action, color = "STRONG BUY 🚀", "#00FFCC"
                    elif pct > 0: action, color = "BUY 🟢", "#00FFCC"
                    elif pct < -0.8: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "HOLD ⚖️", "#FFA500"

                    # Risk Management (Code Awal)
                    risk = 0.02
                    atr = df['ATR'].iloc[-1]
                    sl = curr - (atr * 1.5) if pct > 0 else curr + (atr * 1.5)
                    tp = curr + (abs(curr-sl) * 2.5)
                    qty = (st.session_state.modal * risk) / abs(curr - sl)

                    # Metrik
                    m1, m2 = st.columns(2); m1.metric("Live Price", f"{curr:,.2f}"); m2.metric(f"AI Target ({horizon_label})", f"{pred:,.2f}", f"{pct:+.2f}%")
                    m3, m4 = st.columns(2); m3.metric("Inst. Flow", inst_flow); m4.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

                    st.markdown(f"""
                        <div style='text-align:center; padding:15px; border:1px solid {color}; border-radius:12px; background:#0a0a0a; margin-bottom:20px;'>
                            <h2 style='color:{color}; margin:0;'>{action}</h2>
                            <p style='color:#FFD700; margin:5px 0;'>Qty: {qty:.4f} Units | SL: {sl:,.2f} | TP: {tp:,.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Charting (Identik Code Awal)
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index[-60:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Upper'], line=dict(color='gray', width=1), name="BB Upper"))
                    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Lower'], line=dict(color='gray', width=1), name="BB Lower"))
                    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, paper_bgcolor='black', plot_bgcolor='black')
                    st.plotly_chart(fig, use_container_width=True)

elif selected == "Radar":
    st.markdown("### 📡 Market Radar Scan")
    radar_kat = st.selectbox("Universe", list(database_aset.keys()))
    if st.button("MULAI SCANNING"):
        results = []
        progress = st.progress(0)
        assets = database_aset[radar_kat]
        for idx, ticker in enumerate(assets):
            progress.progress((idx + 1) / len(assets))
            df_r, pred_r, _ = train_ai_pro(ticker, interval_map[horizon_label], period_map[horizon_label], 5)
            if df_r is not None:
                curr_r = df_r['Close'].iloc[-1]
                pct_r = ((pred_r - curr_r) / curr_r) * 100
                results.append({"Aset": ticker, "Price": round(curr_r, 4), "Forecast %": f"{pct_r:+.2f}%", "Signal": "BUY" if pct_r > 0 else "SELL"})
        st.table(pd.DataFrame(results))

else:
    st.title("⚙️ Settings")
    st.session_state.epochs = st.slider("Model Accuracy (Epochs)", 10, 50, 15)
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









































