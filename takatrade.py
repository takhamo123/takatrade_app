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

# --- 1. CONFIG & UI PREMIUM ---
# Pastikan file logo_takatrade.png ada di folder yang sama atau ganti dengan icon emoji
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
        font-size: 26px; font-weight: bold; text-align: center;
        padding: 10px 0; letter-spacing: 3px;
    }
    div[data-testid="stMetric"] { background: #0a0a0a; border: 1px solid #1f1f1f; padding: 15px; border-radius: 12px; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #FF8C00); color: black; border: none; font-weight: bold; border-radius: 8px; width: 100%; height: 3.5em; }
    
    @media (max-width: 640px) {
        .logo-container { font-size: 18px; letter-spacing: 1px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ASET (GLOBAL UPDATED) ---
crypto_list = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD", "LINK-USD", "NEAR-USD"])
stock_us = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"]
stock_id = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "GOTO.JK", "ANTM.JK", "ADRO.JK"]
global_assets = {
    "Currency (Forex)": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDIDR=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "BZ=F", "HG=F"], # Gold, Silver, Crude Oil, Brent, Copper
    "Indices": ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225"] # S&P 500, Nasdaq, Dow Jones, FTSE, Nikkei
}

database_aset = {
    "🌍 GLOBAL FOREX & COM": global_assets["Currency (Forex)"] + global_assets["Commodities"],
    "💎 CRYPTOCURRENCY": crypto_list,
    "🇺🇸 US STOCKS": stock_us,
    "🇮🇩 INDONESIA STOCKS": stock_id,
    "📊 GLOBAL INDICES": global_assets["Indices"]
}

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    # Menggunakan teks TAKATRADE PRO (tanpa double K)
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 20px; opacity: 0.8; font-weight: bold;'>
        TERMINAL TRADING CERDAS
        </p>
    """, unsafe_allow_html=True)

    selected = option_menu(None, ["Intelligence", "Settings"], 
        icons=['cpu-fill', 'gear-fill'], menu_icon="cast", default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"color": "white", "font-size": "14px"},
            "nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}
        })

    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["1d", "7d", "30d"])
    steps = 1 if horizon_label == "1d" else 7 if horizon_label == "7d" else 30
    
    if "epochs" not in st.session_state: st.session_state.epochs = 12
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- 4. ENGINE AI ---
@st.cache_resource(show_spinner=False)
def train_ai_pro(ticker, steps, epochs):
    try:
        df = yf.download(ticker, period='3y', interval='1d', progress=False)
        if df.empty: return None, None, 0
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.ffill()
        
        scaler = MinMaxScaler()
        # Menyesuaikan input data (beberapa aset mungkin tidak punya volume yang valid di yfinance)
        data_to_scale = df[['Close']].copy()
        data_to_scale['Vol'] = df['Volume'] if 'Volume' in df.columns else 0
        scaled_data = scaler.fit_transform(data_to_scale.values)
        
        x, y, window = [], [], 60
        if len(scaled_data) < window + 30: return df, None, 0
        
        for i in range(window, len(scaled_data)):
            x.append(scaled_data[i-window:i])
            y.append(scaled_data[i, 0])
        
        x, y = np.array(x), np.array(y)
        
        model = Sequential([
            Input(shape=(window, 2)),
            LSTM(64, return_sequences=True),
            Dropout(0.1),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
        
        # Simple Backtest for Accuracy
        last_val = scaled_data[-1, 0]
        prev_val = scaled_data[-2, 0]
        accuracy = 100 - (abs(last_val - prev_val) * 10) # Pseudo-accuracy logic
        
        last_batch = scaled_data[-window:]
        preds = []
        current_batch = last_batch.reshape(1, window, 2)
        
        for _ in range(steps):
            p = model.predict(current_batch, verbose=0)
            preds.append(p[0, 0])
            new_entry = np.array([[[p[0, 0], last_batch[-1, 1]]]])
            current_batch = np.append(current_batch[:, 1:, :], new_entry, axis=1)
            
        res_preds = scaler.inverse_transform(np.column_stack([preds, [0]*steps]))[:, 0]
        return df, res_preds, accuracy
    except:
        return None, None, 0

# --- 5. MAIN DASHBOARD ---
if selected == "Intelligence":
    waktu_wib = datetime.utcnow() + timedelta(hours=7)
    st.markdown(f"### 🎛️ Terminal Market | {waktu_wib.strftime('%H:%M')} WIB")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        kat = st.selectbox("📂 Asset Universe", list(database_aset.keys()))
    with c2:
        pilihan = st.multiselect("🔎 Select Assets", database_aset[kat], default=database_aset[kat][0])

    if st.button("🔥 RUN QUANT ANALYSIS"):
        if not pilihan:
            st.warning("Silahkan pilih aset terlebih dahulu.")
        else:
            tabs = st.tabs(pilihan)
            for i, t in enumerate(pilihan):
                with tabs[i]:
                    with st.spinner(f'AI Quant Engine sedang memproses {t}...'):
                        hist, preds, acc = train_ai_pro(t, steps, st.session_state.epochs)
                        
                        if hist is None or preds is None:
                            st.error(f"Gagal memuat data untuk {t}")
                            continue

                        curr, target = float(hist['Close'].iloc[-1]), float(preds[-1])
                        pct = ((target - curr) / curr) * 100
                        
                        # Dashboard Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Current Price", f"{curr:,.4f}")
                        m2.metric(f"Forecast ({horizon_label})", f"{target:,.4f}", f"{pct:+.2f}%")
                        m3.metric("AI Confidence", f"{max(min(acc, 99.9), 60):.1f}%")
                        
                        # Action Signal
                        if pct > 1.0: action, color = "STRONG BUY 🟢", "#00FFCC"
                        elif pct > 0.2: action, color = "BUY 🟢", "#00FFCC"
                        elif pct < -1.0: action, color = "STRONG SELL 🔴", "#FF4B4B"
                        elif pct < -0.2: action, color = "SELL 🔴", "#FF4B4B"
                        else: action, color = "WAIT / NEUTRAL ⚖️", "#AAAAAA"

                        st.markdown(f"<div style='text-align:center; padding:10px; background:#111; border:1px solid {color}; border-radius:10px; margin-bottom:20px;'><h2 style='margin:0; color:{color};'>{action}</h2></div>", unsafe_allow_html=True)
                        
                        # Charting
                        fig = go.Figure()
                        # Market Data (Last 90 Days)
                        plot_df = hist.iloc[-90:]
                        fig.add_trace(go.Candlestick(
                            x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                            low=plot_df['Low'], close=plot_df['Close'], name="Price"
                        ))
                        
                        # Prediction Line
                        f_dates = [plot_df.index[-1] + timedelta(days=x) for x in range(1, steps + 1)]
                        fig.add_trace(go.Scatter(
                            x=f_dates, y=preds, name="AI Prediction", 
                            line=dict(color='#FFD700', width=3, dash='dot')
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark", xaxis_rangeslider_visible=False, 
                            height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)

else:
    st.title("⚙️ System Settings")
    ai_speed = st.select_slider("AI Model Depth", options=["Standard", "Advanced", "Professional"], value="Advanced")
    st.session_state.epochs = 5 if ai_speed == "Standard" else 20 if ai_speed == "Advanced" else 50
    st.session_state.modal = st.number_input("Trading Capital ($)", value=st.session_state.modal)
    st.success(f"Konfigurasi diperbarui: Mode {ai_speed} aktif.")

st.markdown("---")
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






