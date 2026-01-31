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

# Inisialisasi Resource NLP
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. CONFIG & UI PREMIUM ---
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .stApp { background-color: #000000; color: #ffffff; font-family: 'JetBrains Mono', monospace; }
    
    /* Sidebar Obsidian Style */
    section[data-testid="stSidebar"] { 
        background-color: #050505 !important; 
        border-right: 1px solid #1a1a1a; 
    }
    
    /* Gold Gradient Logo */
    .logo-container {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 26px; font-weight: 800; text-align: center;
        padding: 20px 0; letter-spacing: 4px;
        border-bottom: 1px solid #1a1a1a;
        margin-bottom: 20px;
    }
    
    /* Futuristic Metric Card */
    div[data-testid="stMetric"] { 
        background: #0a0a0a; 
        border: 1px solid #222; 
        padding: 20px; 
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Premium Button */
    .stButton>button { 
        background: linear-gradient(45deg, #FFD700, #FF8C00); 
        color: #000; border: none; font-weight: 800; 
        border-radius: 10px; width: 100%; height: 3.8em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3); }
    
    /* News Card */
    .news-card { 
        background: #0d0d0d; 
        border-left: 4px solid #FFD700; 
        padding: 15px; 
        margin-bottom: 12px; 
        border-radius: 8px;
        border-right: 1px solid #222;
        border-top: 1px solid #222;
    }
    
    /* Signal Box */
    .status-box {
        text-align:center; padding:20px; 
        background: rgba(10,10,10,0.8); 
        border-radius:15px; 
        margin-bottom:25px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. UNIVERSAL ASSET DATABASE ---
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
    
    selected = option_menu(None, ["Intelligence", "Radar", "Settings"], 
        icons=['cpu-fill', 'broadcast', 'gear-fill'], default_index=0, 
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"color": "white", "font-size": "14px", "text-align": "left", "margin":"5px"},
            "nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "800"}
        })
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("⏱️ Timeframe Settings")
    horizon_label = st.select_slider("Select Horizon", options=["5m", "15m", "1h", "1d"], value="1h")
    
    # Logic Horizon
    interval_map = {"5m":"5m", "15m":"15m", "1h":"60m", "1d":"1d"}
    period_map = {"5m":"1d", "15m":"5d", "1h":"1mo", "1d":"2y"}
    
    if "epochs" not in st.session_state: st.session_state.epochs = 15

# --- 4. CORE AI ENGINE (PRECISION DUAL-PATH) ---
def add_indicators(df):
    df = df.copy()
    # High Precision RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df.ffill().bfill().dropna()

def train_ai_engine(ticker, interval, period, epochs):
    K.clear_session()
    gc.collect()
    try:
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
        
        model = Sequential([
            Input(shape=(60, 3)),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        model.fit(np.array(x), np.array(y), epochs=epochs, batch_size=32, verbose=0)
        
        last_batch = scaled_data[-60:].reshape(1, 60, 3)
        p_scaled = model.predict(last_batch, verbose=0)[0, 0]
        res = scaler.inverse_transform([[p_scaled, 0, 0]])[0, 0]
        return df, res
    except Exception as e:
        return None, None

def get_market_sentiment(ticker):
    try:
        data = yf.Ticker(ticker)
        news = data.news
        if not news: return 0, "NEUTRAL ⚖️"
        scores = [TextBlob(n.get('title', '')).sentiment.polarity for n in news if n.get('title')]
        avg = np.mean(scores) if scores else 0
        label = "POSITIVE 🔥" if avg > 0.05 else "NEGATIVE 📉" if avg < -0.05 else "NEUTRAL ⚖️"
        return avg, label
    except: return 0, "NEUTRAL ⚖️"

# --- 5. MAIN INTERFACE ---
if selected == "Intelligence":
    st.markdown(f"### <span style='color:#FFD700;'>TAKATRADE</span> Intelligence Terminal", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1: current_cat = st.selectbox("📂 Universe", list(database_aset.keys()))
    with c2: selected_assets = st.multiselect("🔎 Focus Assets", database_aset[current_cat], default=database_aset[current_cat][0])

    if st.button("RUN DEEP ANALYSIS"):
        tabs = st.tabs(selected_assets)
        for i, t in enumerate(selected_assets):
            with tabs[i]:
                with st.spinner(f'Synchronizing with Market Data... {t}'):
                    df, pred = train_ai_engine(t, interval_map[horizon_label], period_map[horizon_label], st.session_state.epochs)
                    
                    if df is not None:
                        curr = float(df['Close'].iloc[-1])
                        pct = ((pred - curr) / curr) * 100
                        _, sent_label = get_market_sentiment(t)
                        
                        # Dynamic Color System
                        sig_col = "#00FFCC" if pct > 0.5 and "POSITIVE" in sent_label else "#FF4B4B" if pct < -0.5 and "NEGATIVE" in sent_label else "#FFA500"
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Live Price", f"{curr:,.4f}")
                        m2.metric("AI Target", f"{pred:,.4f}", f"{pct:+.2f}%")
                        m3.metric("Sentimen NLP", sent_label)
                        
                        st.markdown(f"""
                            <div class="status-box" style="border: 2px solid {sig_col}; box-shadow: 0 0 15px {sig_col}33;">
                                <p style="margin:0; font-size: 12px; color:#888;">TRADING SIGNAL</p>
                                <h1 style="margin:0; color:{sig_col}; letter-spacing:3px;">{"STRONG BUY" if pct > 0.5 else "STRONG SELL" if pct < -0.5 else "NEUTRAL"}</h1>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Candle Chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df.index[-60:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            increasing_line_color='#00FFCC', decreasing_line_color='#FF4B4B'
                        )])
                        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # News Expander
                        news_list = yf.Ticker(t).news
                        if news_list:
                            with st.expander("📰 Intelligence Feed"):
                                for n in news_list[:3]:
                                    st.markdown(f"""
                                        <div class="news-card">
                                            <small style="color:#FFD700;">Market Alert</small><br>
                                            <b style="font-size:14px;">{n.get('title', 'Headline')}</b><br>
                                            <a href="{n.get('link', '#')}" target="_blank" style="color:#00FFCC; text-decoration:none; font-size:11px;">View Source →</a>
                                        </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error(f"Analysis failed for {t}. Check market availability.")

elif selected == "Radar":
    st.markdown("### 📡 Multi-Asset Signal Radar")
    radar_category = st.selectbox("Select Scanning Sector", list(database_aset.keys()))
    
    if st.button("INITIATE SECTOR SCAN"):
        results = []
        bar = st.progress(0)
        target_assets = database_aset[radar_category]
        
        for idx, t in enumerate(target_assets):
            bar.progress((idx + 1) / len(target_assets))
            # Scan with lower epochs for speed
            df_r, pred_r = train_ai_engine(t, interval_map[horizon_label], period_map[horizon_label], 5)
            
            if df_r is not None:
                curr_r = float(df_r['Close'].iloc[-1])
                pct_r = ((pred_r - curr_r) / curr_r) * 100
                sig = "BUY" if pct_r > 0.5 else "SELL" if pct_r < -0.5 else "HOLD"
                results.append({"Asset": t, "Price": round(curr_r, 4), "Growth %": f"{pct_r:+.2f}%", "Signal": sig})
        
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.applymap(lambda x: 'color: #00FFCC' if x == 'BUY' else ('color: #FF4B4B' if x == 'SELL' else 'color: #FFA500'), subset=['Signal']), use_container_width=True)
        else:
            st.warning("Radar scanning produced no data.")

else:
    st.title("⚙️ AI Terminal Settings")
    st.session_state.epochs = st.select_slider("AI Training Precision", options=[5, 15, 30, 50], value=15)
    st.info("Higher epochs improve accuracy but increase processing time.")
    st.markdown("---")
    st.caption("TAKATRADE PRO v3.0 | 2026 Precise Trading Terminal")

st.caption("© 2026 TAKATRADE PRO | Obsidian Edition")

# Instal
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.


































