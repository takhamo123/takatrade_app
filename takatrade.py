import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import backend as K # Untuk reset session RAM
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import gc # Untuk paksa hapus data dari RAM

# --- 1. CONFIG & UI PREMIUM ---
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

# --- 2. DATABASE ASET ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "BCH-USD", "SHIB-USD", "NEAR-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD", "TIA-USD", "SEI-USD", "INJ-USD", "STX-USD", "ALGO-USD", "FTM-USD", "EGLD-USD", "ATOM-USD", "HBAR-USD", "IMX-USD", "FET-USD", "RENDER-USD", "TAO-USD", "RNDR-USD", "AKASH-USD", "ONDO-USD", "PENDLE-USD", "UNI-USD", "AAVE-USD", "LDO-USD", "MKR-USD", "RUNE-USD", "JUP-USD", "CAKE-USD", "OKB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD", "POPCAT-USD", "BRETT-USD", "MOG-USD"])
global_indices_forex = sorted(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURCHF=X", "CHFJPY=X", "EURAUD=X", "GBPAUD=X", "CADJPY=X", "NZDJPY=X", "AUDNZD=X", "USDIDR=X", "SGDIDR=X", "USDSGD=X", "USDTHB=X", "USDHKD=X", "USDCNY=X", "USDMXN=X", "USDMYR=X", "USDPHP=X", "USDVND=X", "USDKRW=X", "^JKSE", "^GSPC", "^IXIC", "^DJI", "^N225", "^HSI", "^FTSE", "^GDAXI", "^FCHI", "^AXJO", "^STI", "GC=F", "SI=F", "CL=F", "BZ=F", "HG=F", "NG=F", "PA=F", "PL=F"])
stock_us = sorted(["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "COIN", "JPM", "V"])
stock_id = sorted(["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK", "ADRO.JK", "BBNI.JK", "UNVR.JK", "BRMS.JK"])

database_aset = {"🌍 GLOBAL MARKET & FOREX": global_indices_forex, "💎 CRYPTOCURRENCY": crypto_list, "🇺🇸 US STOCKS": stock_us, "🇮🇩 INDONESIA STOCKS": stock_id}

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 25px; opacity: 0.85; font-weight: bold;'>TERMINAL TRADING CERDAS</p>", unsafe_allow_html=True)

    selected = option_menu(None, ["Intelligence", "Settings"], 
        icons=['cpu-fill', 'gear-fill'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}})

    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "10m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    
    interval_map = {"5m":"5m", "10m":"2m", "15m":"15m", "30m":"30m", "1h":"60m", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}
    period_map = {"5m":"1d", "10m":"1d", "15m":"5d", "30m":"5d", "1h":"1mo", "1d":"3y", "1wk":"max", "1mo":"max"}
    
    steps = 1 
    if "epochs" not in st.session_state: st.session_state.epochs = 12
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- 4. ENGINE AI (MEMORY OPTIMIZED) ---
def train_ai_pro(ticker, interval, period, steps, epochs):
    # Tahap 1: Bersihkan sisa memori sebelum proses dimulai
    K.clear_session()
    gc.collect()

    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty: return None, None, 0
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.ffill()
    
    if 'Volume' not in df.columns or df['Volume'].isna().all():
        df['Volume'] = 0
        
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close', 'Volume']].values)
    
    x, y, window = [], [], 60
    if len(scaled_data) <= window: window = len(scaled_data) // 2
    
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
    
    # BACKTESTING 10 candles
    test_len = 10 if len(scaled_data) > 10 else 1
    test_batch = scaled_data[-(window+test_len):-test_len]
    bt_preds = []
    for _ in range(test_len):
        p = model.predict(test_batch.reshape(1, window, 2), verbose=0)
        bt_preds.append(p[0, 0])
        new_entry = np.array([[p[0, 0], scaled_data[-test_len+len(bt_preds)-1, 1]]])
        test_batch = np.append(test_batch[1:], new_entry, axis=0)
    
    accuracy = 100 - (np.mean(np.abs(scaled_data[-test_len:, 0] - np.array(bt_preds))) * 100)
    
    # PREDIKSI MASA DEPAN
    last_batch = scaled_data[-window:].tolist()
    preds = []
    for _ in range(steps):
        p = model.predict(np.array(last_batch[-window:]).reshape(1, window, 2), verbose=0)
        preds.append(p[0, 0])
        last_batch.append([p[0, 0], last_batch[-1][1]])
        
    res_preds = scaler.inverse_transform(np.column_stack([preds, [0]*steps]))[:, 0]
    
    # Tahap 2: PENGHAPUSAN MEMORI KRUSIAL
    del model # Hapus variabel model
    K.clear_session() # Reset session backend TensorFlow
    gc.collect() # Paksa Python membuang sampah memori ke sistem
    
    return df, res_preds, accuracy

# --- 5. MAIN DASHBOARD ---
if selected == "Intelligence":
    waktu_wib = datetime.utcnow() + timedelta(hours=7)
    st.markdown(f"### TAKATRADE Pro | {waktu_wib.strftime('%H:%M')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        kat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2:
        pilihan = st.multiselect("🔎 Aset", database_aset[kat], default=database_aset[kat][0])

    if st.button("EXECUTE"):
        tabs = st.tabs(pilihan)
        for i, t in enumerate(pilihan):
            with tabs[i]:
                with st.spinner(f'AI memproses {t}...'):
                    df_raw, preds, acc = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], steps, st.session_state.epochs)
                    
                    if df_raw is None:
                        st.error(f"Gagal mengambil data untuk {t}.")
                        continue
                        
                    curr, target = float(df_raw['Close'].iloc[-1]), float(preds[-1])
                    pct = ((target - curr) / curr) * 100
                    
                    vol_avg = df_raw['Volume'].rolling(10).mean().iloc[-1]
                    vol_curr = df_raw['Volume'].iloc[-1]
                    vol_chg = (vol_curr / vol_avg) if vol_avg != 0 else 1
                    
                    sentiment = "POSITIVE ✨" if pct > 0 and vol_chg > 1 else "NEGATIVE ⚠️" if pct < 0 else "NEUTRAL ⚖️"
                    if pct > 1.5 and vol_chg > 1: action, color = "STRONG BUY 🟢", "#00FFCC"
                    elif pct > 0: action, color = "BUY 🟢", "#00FFCC"
                    elif pct < -1.5: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "SELL 🔴", "#FF4B4B"

                    m1, m2 = st.columns(2)
                    m1.metric("Price", f"{curr:,.4f}")
                    m2.metric(f"Target ({horizon_label})", f"{target:,.4f}", f"{pct:+.2f}%")
                    m3, m4 = st.columns(2)
                    m3.metric("AI Confidence", f"{acc:.1f}%")
                    m4.metric("Sentiment", sentiment)
                    
                    st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border:1px solid {color}; border-radius:10px; margin-bottom:20px;'><h2 style='margin:0; color:{color};'>{action}</h2></div>", unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df_raw.index[-60:], open=df_raw['Open'].iloc[-60:], high=df_raw['High'].iloc[-60:], low=df_raw['Low'].iloc[-60:], close=df_raw['Close'].iloc[-60:], name="Market"))
                    
                    last_date = df_raw.index[-1]
                    delta_map = {"5m":5, "10m":10, "15m":15, "30m":30, "1h":60, "1d":1440, "1wk":10080, "1mo":43200}
                    f_dates = [last_date + timedelta(minutes=delta_map[horizon_label] * (x+1)) for x in range(steps)]
                    
                    fig.add_trace(go.Scatter(x=f_dates, y=preds, name="AI Path", line=dict(color='#FFD700', width=3, dash='dot')))
                    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=5, r=5, t=30, b=5), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"💡 **AI Logic:** Akurasi backtest {acc:.1f}%.")

else:
    st.title("⚙️ Settings")
    ai_speed = st.select_slider("Akurasi Model", options=["Fast", "Balanced", "Precision"], value="Balanced")
    st.session_state.epochs = 5 if ai_speed == "Fast" else 15 if ai_speed == "Balanced" else 30
    st.session_state.modal = st.number_input("Modal Investasi ($)", value=1000)

st.caption("TAKATRADE PRO © 2026 | Terminal Trading Cerdas")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.

























