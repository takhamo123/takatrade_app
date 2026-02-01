# ==============================================================================
# 1. IMPORTS & SETUP AWAL
# ==============================================================================

# Import Library Core & UI
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import Library Data & Numerics
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import gc

# Import Library AI & Machine Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras import backend as K
from sklearn.preprocessing import RobustScaler

# Import Library Analisis Sentimen
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Setup Keamanan & Konfigurasi Awal ---
# Muat variabel dari file .env untuk keamanan
load_dotenv()

# Unduh data NLTK yang diperlukan. 'quiet=True' mencegah output yang tidak perlu.
nltk.download('vader_lexicon', quiet=True)

# Hilangkan warning pandas
pd.options.mode.chained_assignment = None


# ==============================================================================
# 2. KONFIGURASI UI & DATABASE ASET
# ==============================================================================

# --- Konfigurasi Halaman & Gaya Visual ---
st.set_page_config(page_title="TAKATRADE PRO", layout="wide", page_icon="logo_takatrade.png")

# Refresh otomatis setiap 30 menit
st.markdown('<meta http-equiv="refresh" content="1800">', unsafe_allow_html=True)

# CSS Kustom untuk tampilan premium
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
    .news-card { background: #0a0a0a; border-left: 3px solid #FFD700; padding: 10px; margin-bottom: 10px; border-radius: 4px; border-right: 1px solid #1a1a1a; }
    @media (max-width: 640px) {
        .logo-container { font-size: 20px; letter-spacing: 2px; }
        div[data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Database Aset ---
# Daftar aset telah diperluas dengan lebih banyak pilihan

crypto_list = sorted([
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "DOT-USD", "XRP-USD", "ADA-USD", 
    "LINK-USD", "MATIC-USD", "UNI-USD", "AAVE-USD", "ATOM-USD", "AR-USD", "OP-USD", "LDO-USD", "INJ-USD"
])

global_indices_forex = sorted([
    "GC=F", "SI=F", "CL=F", "NG=F",  # Tambah: Natural Gas
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDIDR=X", "AUDUSD=X", "NZDUSD=X", # Tambah: AUD/USD, NZD/USD
    "^GSPC", "^IXIC", "^DJI", "^JKSE", "^STOXX50E" # Tambah: European Index
])

stock_us = sorted([
    "NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "AMD", "NFLX", "COIN",
    "JPM", "BAC", "WMT", "JNJ", "UNH", "PG", "KO", "XOM", "CVX", "MA" # Tambah: Finance, Retail, Healthcare, Energy, dll.
])

stock_id = sorted([
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BTPN.JK", # Tambah: BTPN
    "TLKM.JK", "ISAT.JK", "EXCL.JK", # Tambah: Telkom lainnya
    "ASII.JK", "ANTM.JK", "ADRO.JK", "BRMS.JK", "GOTO.JK",
    "UNVR.JK", "INDF.JK", "KLBF.JK", "BBRI.JK" # Tambah: Consumer Goods, Pharma
])

database_aset = {
    "🌍 GLOBAL MARKET & FOREX": global_indices_forex,
    "💎 CRYPTOCURRENCY": crypto_list,
    "🇺🇸 US STOCKS": stock_us,
    "🇮🇩 INDONESIA STOCKS": stock_id
}


# ==============================================================================
# 3. FUNGSI UTILITAS (TELEGRAM & SENTIMEN)
# ==============================================================================

def send_telegram_alert(message):
    """Mengirim notifikasi ke Telegram jika kredensial sudah diatur."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id or token == "YOUR_TOKEN_HERE" or chat_id == "YOUR_CHAT_ID_HERE":
        # Tidak menampilkan warning di sini agar tidak mengganggu saat radar scan
        # st.warning("Telegram Alert tidak aktif. Atur TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di file .env")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        # Tidak menampilkan error di sini agar tidak mengganggu
        # st.error(f"Gagal mengirim notifikasi Telegram: {e}")
        pass

def get_sentiment(text):
    """Menganalisis sentimen dari sebuah teks dan mengembalikan label serta skor."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)
    score = sentiment_dict['compound']
    
    if score >= 0.05:
        return "Positive 🟢", score
    elif score <= -0.05:
        return "Negative 🔴", score
    else:
        return "Neutral ⚪", score


# ==============================================================================
# 4. MESIN INTI (DATA & AI)
# ==============================================================================

def add_indicators_clean(df):
    """Menambahkan indikator teknis dan membersihkan data."""
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/loss).replace(0, np.nan)))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    return df.ffill().bfill().dropna()

@st.cache_data(ttl=300)
def fetch_market_data(ticker, period, interval):
    """Mengambil data pasar dari Yahoo Finance dengan cache."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if data.empty:
            st.error(f"Tidak dapat menemukan data untuk ticker '{ticker}'. Periksa kembali simbol ticker.")
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data untuk '{ticker}'. Error: {e}. Periksa koneksi internet Anda.")
        return pd.DataFrame()

def train_ai_pro(ticker, interval, period, epochs):
    """
    Melatih model LSTM untuk prediksi harga.
    Model selalu dilatih ulang setiap kali fungsi dipanggil (sesuai permintaan).
    """
    K.clear_session()
    gc.collect()
    
    df_raw = fetch_market_data(ticker, period, interval)
    if df_raw.empty or len(df_raw) < 60:
        return None, 0
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    df_clean = add_indicators_clean(df_raw)

    # --- PERBAIKAN KRUSIAL: PEMERIKSAAN DATA SETELAH PEMBERSIHAN ---
    # Pastikan kita punya cukup data SETELAH dibersihkan sebelum melatih model
    if df_clean.empty or len(df_clean) < 60:
        st.warning(f"Data untuk {ticker} tidak mencukupi setelah dibersihkan. Coba timeframe/periode yang lebih besar.")
        return None, 0

    features = ['Close', 'Volume', 'RSI', 'MACD']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_clean[features].values)
    
    window = 30
    x, y = [], []
    for i in range(window, len(scaled_data)):
        x.append(scaled_data[i-window:i])
        y.append(scaled_data[i, 0])
    
    model = Sequential([
        Input(shape=(window, 4)),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.1),
        Dense(25, activation='swish'),
        Dense(1)
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    model.fit(np.array(x), np.array(y), epochs=epochs, batch_size=32, verbose=0, callbacks=[callback])
    
    last_batch = scaled_data[-window:].reshape(1, window, 4)
    pred_scaled = model.predict(last_batch, verbose=0)
    res_pred = scaler.inverse_transform([[pred_scaled[0,0], 0, 0, 0]])[0,0]
    
    return df_clean, res_pred


# ==============================================================================
# 5. FUNGSI TAMBAHAN (BERITA)
# ==============================================================================

def get_news_aggregator(ticker):
    """Mengambil berita terkini dan menganalisis sentimennya."""
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news
        if not raw_news:
            return []
        
        processed_news = []
        for n in raw_news[:3]:
            title = n.get('title') or "Intelligence Update"
            publisher = n.get('publisher') or "Unknown Source"
            sentiment, score = get_sentiment(title)
            
            processed_news.append({
                'title': title, 
                'publisher': publisher,
                'sentiment': sentiment, 
                'score': score
            })
        return processed_news
    except Exception as e:
        st.caption(f"Tidak bisa mengambil berita untuk {ticker}: {e}")
        return []


# ==============================================================================
# 6. DASHBOARD UTAMA APLIKASI
# ==============================================================================

# --- Navigasi Sidebar ---
with st.sidebar:
    st.markdown('<div class="logo-container">TAKATRADE PRO</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #FFD700; font-family: sans-serif; font-size: 10px; letter-spacing: 2px; margin-top: -15px; margin-bottom: 25px; opacity: 0.85; font-weight: bold;'>TERMINAL TRADING CERDAS</p>", unsafe_allow_html=True)
    
    selected = option_menu(
        None, 
        ["Intelligence", "Radar", "Settings"], 
        icons=['cpu-fill', 'broadcast', 'gear-fill'], 
        default_index=0, 
        styles={"nav-link-selected": {"background-color": "#FFD700", "color": "black", "font-weight": "bold"}}
    )
    
    st.markdown("---")
    st.subheader("⏱️ Forecasting Horizon")
    horizon_label = st.select_slider("Pilih Jangka Waktu", options=["5m", "10m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    interval_map = {"5m":"5m", "10m":"2m", "15m":"15m", "30m":"30m", "1h":"60m", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}
    period_map = {"5m":"1d", "10m":"1d", "15m":"5d", "30m":"5d", "1h":"1mo", "1d":"3y", "1wk":"max", "1mo":"max"}
    
    if "epochs" not in st.session_state: st.session_state.epochs = 15
    if "modal" not in st.session_state: st.session_state.modal = 1000

# --- Tab Intelligence ---
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
                    df_res, pred = train_ai_pro(t, interval_map[horizon_label], period_map[horizon_label], st.session_state.epochs)
                    
                    if df_res is None or pred == 0:
                        st.warning(f"Data {t} tidak mencukupi. Coba timeframe/periode lebih besar.")
                        continue
                    
                    curr = df_res['Close'].iloc[-1]
                    pct = ((pred - curr) / curr) * 100
                    macd_bull = df_res['MACD'].iloc[-1] > df_res['Signal'].iloc[-1]
                    rsi_now = df_res['RSI'].iloc[-1]
                    inst_flow = "ACCUMULATION 🏦" if df_res['OBV'].iloc[-1] > df_res['OBV'].rolling(10).mean().iloc[-1] else "DISTRIBUTION 🏛️"
                    
                    if pct > 0.6 and macd_bull: action, color = "STRONG BUY 🚀", "#00FFCC"
                    elif pct > 0: action, color = "BUY 🟢", "#00FFCC"
                    elif pct < -0.6: action, color = "STRONG SELL 🔴", "#FF4B4B"
                    else: action, color = "HOLD ⚖️", "#FFA500"

                    if "STRONG" in action:
                        alert_msg = f"🚀 *TAKATRADE ALERT*\nAset: {t}\nSinyal: {action}\nHarga: {curr:,.2f}\nTarget: {pred:,.2f} ({pct:+.2f}%)"
                        send_telegram_alert(alert_msg)

                    atr = df_res['ATR'].iloc[-1]
                    sl = curr - (atr * 2.0) if "BUY" in action else curr + (atr * 2.0)
                    tp = curr + (abs(curr-sl) * 2.5)
                    qty = (st.session_state.modal * 0.02) / abs(curr - sl if curr != sl else 1)

                    m1, m2 = st.columns(2); m1.metric("Live Price", f"{curr:,.4f}"); m2.metric(f"AI Target ({horizon_label})", f"{pred:,.4f}", f"{pct:+.2f}%")
                    m3, m4 = st.columns(2); m3.metric("Inst. Flow", inst_flow); m4.metric("RSI (14)", f"{rsi_now:.2f}")

                    st.markdown(f"""
                        <div style='text-align:center; padding:15px; background:#0a0a0a; border:1px solid {color}; border-radius:12px; margin-bottom:20px;'>
                            <h2 style='margin:0; color:{color};'>{action}</h2>
                            <p style='margin:5px 0; color:#FFD700; font-weight:bold;'>OPTIMIZED QTY: ${qty:.2f}</p>
                            <p style='margin:0; color:gray; font-size:12px;'>TP: {tp:,.4f} | SL: {sl:,.4f} | RR 1:2.5</p>
                        </div>
                    """, unsafe_allow_html=True)

                    news = get_news_aggregator(t)
                    if news:
                        with st.expander("📰 Latest Market Intelligence & Sentiment"):
                            for n in news:
                                st.markdown(f"""
                                    <div class='news-card'>
                                        <b>{n['title']}</b>
                                        <br><small style='color: #888;'>Sumber: {n['publisher']}</small>
                                        <br><small style='color: #FFD700; font-weight: bold;'>Sentiment: {n['sentiment']} (Score: {n['score']:.2f})</small>
                                    </div>
                                """, unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df_res.index[-60:], open=df_res['Open'], high=df_res['High'], low=df_res['Low'], close=df_res['Close'], name="Market", increasing_line_color='#00FFCC', decreasing_line_color='#FF4B4B'))
                    fig.add_trace(go.Scatter(x=df_res.index[-60:], y=df_res['Upper'], line=dict(color='rgba(255,215,0,0.3)', width=1.5), name="BB Upper"))
                    fig.add_trace(go.Scatter(x=df_res.index[-60:], y=df_res['Lower'], line=dict(color='rgba(255,215,0,0.3)', width=1.5), fill='tonexty', fillcolor='rgba(255,215,0,0.03)', name="BB Lower"))
                    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550, paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Price Action", side="right"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Backtesting Module ---
                    st.markdown("---")
                    with st.expander("📊 AI Performance Backtest (Walk-Forward Analysis)"):
                        with st.spinner("Validating historical accuracy..."):
                            hist_data = df_res.copy()
                            features_bt = ['Close', 'Volume', 'RSI', 'MACD']
                            scaler_bt = RobustScaler()
                            win_count, total_tests, window_bt = 0, 0, 30
                            
                            for i in range(len(hist_data) - window_bt - 5):
                                train_data = hist_data.iloc[i : i + window_bt]
                                if len(train_data) < window_bt: continue
                                
                                scaled_train = scaler_bt.fit_transform(train_data[features_bt].values)
                                x_train = np.array([scaled_train[j-window_bt:j] for j in range(window_bt, len(scaled_train))])
                                y_train = np.array([scaled_train[j, 0] for j in range(window_bt, len(scaled_train))])

                                if len(x_train) > 0:
                                    mini_model = Sequential([Input(shape=(window_bt, 4)), LSTM(50, activation='relu'), Dense(1)])
                                    mini_model.compile(optimizer='adam', loss='huber')
                                    mini_model.fit(x_train, y_train, epochs=5, verbose=0)
                                    
                                    last_batch_scaled = scaled_train[-window_bt:].reshape(1, window_bt, 4)
                                    pred_scaled = mini_model.predict(last_batch_scaled, verbose=0)
                                    pred_price = scaler_bt.inverse_transform([[pred_scaled[0,0], 0,0,0]])[0,0]
                                    
                                    actual_future_price = hist_data['Close'].iloc[i + window_bt + 5]
                                    
                                    if (pred_price > train_data['Close'].iloc[-1] and actual_future_price > train_data['Close'].iloc[-1]) or \
                                       (pred_price < train_data['Close'].iloc[-1] and actual_future_price < train_data['Close'].iloc[-1]):
                                        win_count += 1
                                    total_tests += 1
                                    K.clear_session()

                            win_rate = (win_count / total_tests) * 100 if total_tests > 0 else 0
                            c_bt1, c_bt2, c_bt3 = st.columns(3)
                            c_bt1.metric("Historical Win Rate", f"{win_rate:.1f}%")
                            c_bt2.metric("Predictive Alpha", f"{(win_rate - 50) * 0.1:+.2f}")
                            c_bt3.metric("Model Stability", "EXCELLENT" if win_rate > 60 else "STABLE")
                            st.caption("Analisis pembanding arah prediksi AI dengan data historis menggunakan simulasi walk-forward. Komputasi intensif.")

# --- Tab Radar ---
elif selected == "Radar":
    st.markdown("### 📡 Market Radar Scan")
    radar_kat = st.selectbox("Universe", list(database_aset.keys()))
    if st.button("MULAI SCANNING"):
        results = []
        progress = st.progress(0)
        assets = database_aset[radar_kat]
        for idx, ticker in enumerate(assets):
            progress.progress((idx + 1) / len(assets))
            df_r, pred_r = train_ai_pro(ticker, interval_map[horizon_label], period_map[horizon_label], 10)
            if df_r is not None:
                curr_r = df_r['Close'].iloc[-1]
                pct_r = ((pred_r - curr_r) / curr_r) * 100
                sig = "BUY" if pct_r > 0.2 else "SELL" if pct_r < -0.2 else "HOLD"
                results.append({"Aset": ticker, "Price": round(curr_r, 4), "Forecast": f"{pct_r:+.2f}%", "Signal": sig})
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        if results:
            report = "📡 *RADAR SCAN REPORT*\n" + "\n".join([f"- {r['Aset']}: {r['Signal']} ({r['Forecast']})" for r in results[:10]])
            send_telegram_alert(report)

# --- Tab Settings ---
else:
    st.title("⚙️ Settings")
    st.session_state.epochs = st.slider("Model Precision (Epochs)", 10, 60, 20)
    st.session_state.modal = st.number_input("Trading Capital ($)", value=1000)

# Footer
st.caption("TAKATRADE PRO © 2026 | Terminal Trading Cerdas Berbasis Deep Learning")

# Install
# pip install streamlit yfinance pandas pandas_ta numpy scikit-learn tensorflow plotly streamlit-option-menu scipy
# pip install streamlit yfinance pandas numpy pandas_ta scikit-learn tensorflow plotly streamlit-option-menu
# Membuka Terminal : Ctrl + J
# Jalankan perintah : streamlit run takatrader.py
# Menutup Terminal : Ctrl + C
# Waktu Tunggu Training: Karena sistem ini menggunakan Deep Learning (LSTM), proses "Analisa Quant" akan memakan waktu 30-60 detik per aset (tergantung spesifikasi komputer Anda). Jangan menutup aplikasi saat proses ini berjalan.
# Kualitas Koneksi: Data ditarik secara real-time dari Yahoo Finance. Pastikan koneksi internet stabil agar proses download data tidak terputus di tengah jalan.

# Akurasi Bukan Kepastian: Ingat, skor AI Confidence yang muncul adalah cerminan masa lalu. Jika skornya rendah (di bawah 70%), sebaiknya jangan mengambil keputusan hanya berdasarkan AI tersebut.






















































