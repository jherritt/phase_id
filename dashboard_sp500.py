import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import json
import os
from functools import lru_cache
import plotly.graph_objects as go

# ── Page Config (must be first Streamlit command) ─────────────────────────────
st.set_page_config(page_title="Phase_ID.ai", layout="wide")

# Resolve paths relative to this script file
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# ── Index Selector ─────────────────────────────────────────────────────────────
index_options = ["SP500", "SP400", "NASDAQ", "RUSSELL"]
selected_index = st.sidebar.selectbox("Choose Index:", index_options)
prefix = selected_index.lower()

# ── File Paths ───────────────────────────────────────────────────────────────
#
# ── Data Directory ─────────────────────────────────────────────────────────────
# Check script folder data, then working directory data
script_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
cwd_data = os.path.join(os.getcwd(), "data")
if os.path.isdir(script_data):
    DATA_DIR = script_data
elif os.path.isdir(cwd_data):
    DATA_DIR = cwd_data
else:
    st.error(f"No data directory found. Checked:\n  {script_data}\n  {cwd_data}")
    st.stop()
PH1_PRICE_CSV   = os.path.join(DATA_DIR, f"{prefix}_phase1_price_df.csv")
PH1_MA_PKL      = os.path.join(DATA_DIR, f"{prefix}_phase1_moving_averages.pkl")
PH1_TICKERS_CSV = os.path.join(DATA_DIR, f"{prefix}_phase1_tickers.csv")
PH1_NAMES_CSV   = os.path.join(DATA_DIR, f"{prefix}_phase1_stock_tickers_names.csv")

PH2_PRICE_CSV   = os.path.join(DATA_DIR, f"{prefix}_phase2_price_df.csv")
PH2_MA_PKL      = os.path.join(DATA_DIR, f"{prefix}_phase2_moving_averages.pkl")
PH2_TICKERS_CSV = os.path.join(DATA_DIR, f"{prefix}_phase2_tickers.csv")
PH2_NAMES_CSV   = os.path.join(DATA_DIR, f"{prefix}_phase2_stock_tickers_names.csv")

# ── Page Setup ────────────────────────────────────────────────────────────────
st.title(f"Phase_ID.ai – {selected_index}")
st.markdown(
    """
    Choose a phase and focus on a ticker (or All).  
    • Chart: full 2-year window with 50d & 200d MAs (hover to inspect).  
    • Feedback: flag any ticker whose chart is incorrect.  
    • Selection: pick one ticker or All to view.
    """
)

# ── Utility: normalize datetime index ─────────────────────────────────────────
def normalize_datetime_index(df):
    idx = df.index
    if np.issubdtype(idx.dtype, np.number):
        df.index = pd.to_datetime(idx, unit="s", errors="coerce")
    else:
        df.index = pd.to_datetime(idx, errors="coerce")
    df.sort_index(inplace=True)
    return df

@st.cache_data
def make_chart(df_full, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_full["Date"], y=df_full["Price"], name="Price"))
    fig.add_trace(go.Scatter(x=df_full["Date"], y=df_full["50d MA"], name="50d MA"))
    fig.add_trace(go.Scatter(x=df_full["Date"], y=df_full["200d MA"], name="200d MA"))
    fig.update_layout(title=f"Last 2 Years for {ticker}", height=400, showlegend=True)
    return fig

@st.cache_data
def make_chart_arrays(_dates, prices, ma50, ma200, ticker):
    dates = _dates
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Price"))
    fig.add_trace(go.Scatter(x=dates, y=ma50, name="50d MA"))
    fig.add_trace(go.Scatter(x=dates, y=ma200, name="200d MA"))
    fig.update_layout(
        title=f"{ticker}: Last 2Y",
        height=350,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig

# ── Load Phase Data ───────────────────────────────────────────────────────────
# Phase 1
price_df1 = normalize_datetime_index(pd.read_csv(PH1_PRICE_CSV, index_col=0))
with open(PH1_MA_PKL, "rb") as f:
    moving_averages1 = pickle.load(f)
for df in moving_averages1.values():
    normalize_datetime_index(df)

df_p1 = pd.read_csv(PH1_TICKERS_CSV)
if "Ticker" in df_p1.columns:
    raw_t1 = df_p1["Ticker"]
elif df_p1.shape[1] == 1:
    raw_t1 = df_p1.iloc[:,0]
else:
    st.error("phase1_tickers.csv must have 'Ticker' column or single-column list.")
    st.stop()
tickers1 = raw_t1.dropna().astype(str).tolist()
try:
    df_names1 = pd.read_csv(PH1_NAMES_CSV, index_col=0)
    names1 = df_names1.iloc[:,0].to_dict()
except:
    dfr = pd.read_csv(PH1_NAMES_CSV, header=None, names=["Ticker","Company Name"])
    names1 = dfr.set_index("Ticker")["Company Name"].to_dict()

# Phase 2
price_df2 = normalize_datetime_index(pd.read_csv(PH2_PRICE_CSV, index_col=0))
with open(PH2_MA_PKL, "rb") as f:
    moving_averages2 = pickle.load(f)
for df in moving_averages2.values():
    normalize_datetime_index(df)

df_p2 = pd.read_csv(PH2_TICKERS_CSV)
if "Ticker" in df_p2.columns:
    raw_t2 = df_p2["Ticker"]
elif df_p2.shape[1] == 1:
    raw_t2 = df_p2.iloc[:,0]
else:
    st.error("phase2_tickers.csv must have 'Ticker' column or single-column list.")
    st.stop()
tickers2 = raw_t2.dropna().astype(str).tolist()
try:
    df_names2 = pd.read_csv(PH2_NAMES_CSV, index_col=0)
    names2 = df_names2.iloc[:,0].to_dict()
except:
    dfr = pd.read_csv(PH2_NAMES_CSV, header=None, names=["Ticker","Company Name"])
    names2 = dfr.set_index("Ticker")["Company Name"].to_dict()

# ── Phase Selector ─────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
phase = st.sidebar.selectbox("Choose Phase:", ["Phase 1 - Advancing","Phase 2 - Declining"])
if phase.startswith("Phase 1"):
    price_df, moving_averages, tickers, names = price_df1, moving_averages1, tickers1, names1
    fb_file = os.path.join(DATA_DIR, "feedback_phase1.json")
else:
    price_df, moving_averages, tickers, names = price_df2, moving_averages2, tickers2, names2
    fb_file = os.path.join(DATA_DIR, "feedback_phase2.json")

# load feedback
if os.path.exists(fb_file):
    with open(fb_file, "r") as f: feedback = json.load(f)
else:
    feedback = {}

# ── Ticker Focus via Radio ────────────────────────────────────────────────────
st.sidebar.header("Tickers")
options = ["All"] + tickers
# reset focus if phase changed
def reset_focus():
    st.session_state.focus = "All"
if 'last_phase' not in st.session_state or st.session_state.last_phase != phase:
    reset_focus()
st.session_state.last_phase = phase

focus = st.sidebar.radio("Focus Ticker:", options, index=0)

# set focus (show all charts on one page)
display = [focus] if focus != "All" else tickers
# display all tickers at once
display_tickers = display

# -- Pagination logic (commented out) --
# PAGE_SIZE = 5
# if "page_idx" not in st.session_state: st.session_state.page_idx = 0
# total = len(display)
# pages = max((total-1)//PAGE_SIZE + 1, 1)
# st.session_state.page_idx = min(max(st.session_state.page_idx, 0), pages-1)
# start = st.session_state.page_idx * PAGE_SIZE
# display_tickers = display[start:start+PAGE_SIZE]

# ── Pre-slice recent series for all tickers ─────────────────────────────────────
price_recent = price_df.last("730D")
ma50_recent  = {t: moving_averages["50d"][t].last("730D")  for t in tickers}
ma200_recent = {t: moving_averages["200d"][t].last("730D") for t in tickers}
dates = price_recent.index

# ── Plot & Feedback ──────────────────────────────────────────────────────────
for t in display_tickers:
    cname = names.get(t, "—")
    with st.expander(f"{t} | {cname}", expanded=True):
        # Pull pre-sliced arrays
        prices = price_recent[t].values
        ma50    = ma50_recent[t].values
        ma200   = ma200_recent[t].values
        # Build and render chart
        fig = make_chart_arrays(dates, prices, ma50, ma200, t)
        st.plotly_chart(fig, use_container_width=True)
        # feedback buttons
        flagged = feedback.get(t, False)
        if not flagged:
            if st.button('Flag as Incorrect', key=f'flag_{phase}_{t}'):
                feedback[t] = True
                with open(fb_file,'w') as f: json.dump(feedback,f,indent=2)
                st.success(f"Flagged {t}.")
        else:
            if st.button('Unflag', key=f'unflag_{phase}_{t}'):
                feedback.pop(t,None)
                with open(fb_file,'w') as f: json.dump(feedback,f,indent=2)
                st.success(f"Unflagged {t}.")

# ── Pagination Controls ───────────────────────────────────────────────────────
# if focus == 'All':
#     col_prev, col_page, col_next = st.columns([1,2,1])
#     with col_prev:
#         if st.button('Previous', disabled=st.session_state.page_idx==0):
#             st.session_state.page_idx -=1
#     with col_page:
#         st.markdown(f"**Page {st.session_state.page_idx+1} of {pages}**")
#     with col_next:
#         if st.button('Next', disabled=st.session_state.page_idx>=pages-1):
#             st.session_state.page_idx +=1
