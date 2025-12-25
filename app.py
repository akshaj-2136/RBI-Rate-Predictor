import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import plotly.graph_objects as go
from transformers import pipeline
import nltk
import re
import os
import feedparser
from bs4 import BeautifulSoup

# --- Robust Setup ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

st.set_page_config(layout="wide", page_title="RBI Rate Predictor V1", page_icon="🏦")

# ==========================================
# 🎨 CUSTOM CSS (The "Wix Template" Look)
# ==========================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .sub-text {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.2em;
        margin-bottom: 40px;
    }

    /* Metric Cards (Glassmorphism) */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid #4F46E5;
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2em;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-delta-pos { color: #34D399; font-size: 0.9em; }
    .metric-delta-neg { color: #F87171; font-size: 0.9em; }

    /* Custom Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-weight: 600;
        font-size: 1.1em;
        transition: opacity 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        border: none;
        color: white;
    }
    
    /* Text Area Styling */
    .stTextArea>div>div>textarea {
        background-color: #111827;
        color: #E5E7EB;
        border: 1px solid #374151;
        border-radius: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1F2937;
        padding: 10px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #9CA3AF;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #374151;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 LOGIC & ASSETS
# ==========================================

@st.cache_resource
def load_assets():
    if not os.path.exists('rbi_rate_model_v1.pkl'): return None
    return pickle.load(open('rbi_rate_model_v1.pkl', 'rb'))

@st.cache_resource
def load_nlp():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt", device=-1)

# --- Real Data Fetchers ---
def get_live_oil():
    try:
        oil = yf.Ticker("BZ=F")
        hist = oil.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return 75.0
    except:
        return 75.0

def fetch_rss_feed(url, limit=3):
    try:
        feed = feedparser.parse(url)
        texts = []
        for entry in feed.entries[:limit]:
            raw_text = entry.title + ". " + entry.description
            soup = BeautifulSoup(raw_text, "html.parser")
            clean_text = soup.get_text()
            texts.append(clean_text)
        return " ".join(texts)
    except Exception as e:
        return ""

def get_real_rbi_news():
    url = "https://rbi.org.in/pressreleases_rss.xml"
    text = fetch_rss_feed(url)
    if len(text) < 50:
        return "Inflation remains a priority. The committee decided to keep the policy repo rate unchanged to ensure price stability while supporting growth."
    return text

def get_real_finmin_news():
    url = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3" 
    text = fetch_rss_feed(url)
    if len(text) < 50:
        return "The government is committed to fiscal consolidation and boosting infrastructure spending to drive economic growth in the coming quarters."
    return text

# --- Prediction Engine ---
def safe_sent_tokenize(text):
    try:
        sents = nltk.tokenize.sent_tokenize(text)
        if len(sents)==0: raise ValueError
        return sents
    except:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def get_sentiment_score(text, nlp_pipeline):
    if not text or not text.strip(): return 0.0
    sents = safe_sent_tokenize(text)
    results = []
    for i in range(0, len(sents), 8):
        batch = sents[i:i+8]
        try:
            outs = nlp_pipeline(batch)
            for out in outs:
                label = out['label'].lower()
                score = out['score']
                if 'pos' in label: val = 1.0 * score
                elif 'neg' in label: val = -1.0 * score
                else: val = 0.0
                results.append(val)
        except: pass
    if not results: return 0.0
    return np.mean(results)

def get_prediction(cpi, gdp, oil, rbi_text, finmin_text, model, nlp):
    rbi_score = get_sentiment_score(rbi_text, nlp)
    finmin_score = get_sentiment_score(finmin_text, nlp)
    
    # Logic: FinBERT Pos=1 (Dovish), Neg=-1 (Hawkish).
    final_rbi = -rbi_score 
    final_finmin = finmin_score 
    
    input_data = pd.DataFrame([[cpi, gdp, oil, final_rbi, final_finmin]],
                              columns=['CPI_Inflation', 'GDP_Growth', 'Crude_Oil_Price', 'RBI_Sentiment', 'FinMin_Sentiment'])
    probs = model.predict_proba(input_data)[0]
    return probs, rbi_score, finmin_score

# ==========================================
# 🖥️ MAIN UI LAYOUT
# ==========================================

model = load_assets()
nlp = load_nlp()

# -- Hero Section --
st.markdown('<div class="gradient-text">RBI RATE PREDICTOR V1</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Multi-Class Macro-Economic Forecasting Engine</div>', unsafe_allow_html=True)

if not model:
    st.error("⚠️ Model V1 not found. Please run 'python train_model.py' to generate the brain!")
    st.stop()

# -- Tabs --
tab1, tab2 = st.tabs(["🚀 Live Dashboard", "🎛️ Scenario Simulator"])

# === TAB 1: LIVE DASHBOARD ===
with tab1:
    st.markdown("### 📊 Real-Time Market Conditions")
    
    live_oil = get_live_oil()
    
    # Custom CSS Grid for Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Brent Crude Oil</div>
            <div class="metric-value">${live_oil:.2f}</div>
            <div class="metric-delta-neg">Live Data</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # UPDATED: Using correct Nov 2025 CPI Data (0.71%)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CPI Inflation (Nov '25)</div>
            <div class="metric-value">0.71%</div>
            <div class="metric-delta-pos">Target: 4.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">GDP Growth (Est)</div>
            <div class="metric-value">7.2%</div>
            <div class="metric-delta-pos">Strong</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # News Section
    c_news1, c_news2 = st.columns(2)
    with c_news1:
        st.markdown("##### 🏦 RBI Official Feed (RSS)")
        real_rbi_text = get_real_rbi_news()
        st.text_area("Latest Press Release", value=real_rbi_text, height=150, disabled=True, key="live_rbi")
    with c_news2:
        st.markdown("##### 🏛️ FinMin Official Feed (RSS)")
        real_finmin_text = get_real_finmin_news()
        st.text_area("Latest Government Updates", value=real_finmin_text, height=150, disabled=True, key="live_fin")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate Forecast", type="primary", key="btn_live"):
        with st.spinner("Crunching Macro Data..."):
            # UPDATED: Using 0.71 for CPI
            probs, rbi_s, fin_s = get_prediction(0.71, 7.2, live_oil, real_rbi_text, real_finmin_text, model, nlp)
            
            winner_idx = np.argmax(probs)
            labels = ["RATE CUT", "PAUSE", "RATE HIKE"]
            colors = ["#34D399", "#FBBF24", "#F87171"]
            
            # Result Card
            st.markdown(f"""
            <div style="background: #1F2937; border-radius: 12px; padding: 20px; border-left: 5px solid {colors[winner_idx]}; margin-top: 20px;">
                <h2 style="color: {colors[winner_idx]}; margin:0;">FORECAST: {labels[winner_idx]}</h2>
                <p style="color: #9CA3AF; margin-top: 5px;">
                    Probability Confidence: <strong>{probs[winner_idx]*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Plotly Chart
            fig = go.Figure(data=[go.Bar(
                x=['Rate Cut', 'Pause', 'Rate Hike'],
                y=probs,
                marker_color=[colors[0], colors[1], colors[2]],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='auto',
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E5E7EB'),
                yaxis=dict(showgrid=False),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# === TAB 2: SIMULATOR ===
with tab2:
    st.markdown("### 🎛️ Strategic Scenario Analysis")
    
    c1, c2, c3 = st.columns(3)
    sim_cpi = c1.slider("CPI Inflation (%)", 0.0, 10.0, 0.71) # Updated Range to allow low CPI
    sim_gdp = c2.slider("GDP Growth (%)", 3.0, 10.0, 5.0)
    sim_oil = c3.number_input("Crude Oil Price ($)", value=95.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_txt1, c_txt2 = st.columns(2)
    sim_rbi = c_txt1.text_area("Custom RBI Speech:", "Inflation is unacceptably high. We must act.")
    sim_fin = c_txt2.text_area("Custom FinMin Speech:", "We need to support the recovering economy.")
    
    if st.button("Run Simulation", key="btn_sim"):
        probs, rbi_s, fin_s = get_prediction(sim_cpi, sim_gdp, sim_oil, sim_rbi, sim_fin, model, nlp)
        
        winner_idx = np.argmax(probs)
        labels = ["RATE CUT", "PAUSE", "RATE HIKE"]
        colors = ["#34D399", "#FBBF24", "#F87171"]
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <h1 style="color: {colors[winner_idx]};">{labels[winner_idx]}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Bar(
            x=['Cut', 'Pause', 'Hike'], y=probs,
            marker_color=colors
        )])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E5E7EB'),
            height=200,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)