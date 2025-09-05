import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from BlackScholes import BlackScholes_model
import plotly.graph_objects as go


# Streamlit UI
st.set_page_config(layout="wide") 
st.sidebar.title("Black-Scholes Parameters")
S0 = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.05, min_value=0.0001, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, min_value=0.0001, format="%.4f")
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.005, format="%.4f")

st.sidebar.subheader("Heatmap Parameters")
spot_min = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0, format="%.2f")
spot_max = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0, format="%.2f")
vol_min = st.sidebar.number_input("Min Volatility for Heatmap", value=0.10, step=0.01, min_value=0.0001, format="%.4f")
vol_max = st.sidebar.number_input("Max Volatility for Heatmap", value=0.40, step=0.01, min_value=0.0001, format="%.4f")

spot_steps = st.sidebar.slider("Spot grid size", min_value=6, max_value=25, value=10, step=1)
vol_steps  = st.sidebar.slider("Vol grid size",  min_value=6, max_value=25, value=10, step=1)


# Header 
st.title("Black-Scholes European Option Pricer with no Dividends")
st.latex(r"""
CALL = S_0\,N(d_1) - K e^{-rT} N(d_2), \qquad
PUT = K e^{-rT} N(-d_2) - S_0\,N(-d_1)
""")
st.latex(r"""
d_1 = \frac{\ln\!\left(\frac{S_0}{K}\right)+(r+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}
""")
params_df = pd.DataFrame([{
    "Current Asset Price": S0,
    "Strike Price": K,
    "Time to Maturity (Years)": T,
    "Volatility (σ)": sigma,
    "Risk-Free Rate": r,
}])
st.table(params_df.applymap(lambda v: f"{v:.4f}"))

# PUT/CALL disp
def value_card(title: str, value: float, bg: str, text: str = "#111", border: str = "#00000022"):
    return f"""
    <div style="
        background:{bg};
        color:{text};
        border:1px solid {border};
        border-radius:14px;
        padding:22px 24px;
        height:90px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
    ">
        <div style="font-size:14px; opacity:.85; letter-spacing:.3px;">{title}</div>
        <div style="font-weight:800; font-size:24px; margin-top:6px;">
            ${value:,.2f}
        </div>
    </div>
    """
call_price = BlackScholes_model(S0, K, T, r, sigma, option_type='C')
put_price  = BlackScholes_model(S0, K, T, r, sigma, option_type='P')
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        value_card("CALL Value", call_price, bg="#A7F3D0"),  
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        value_card("PUT Value", put_price, bg="#FECACA"),   
        unsafe_allow_html=True
    )


# heatmaps grid
spot_range = np.linspace(spot_min, spot_max, spot_steps)
vol_range  = np.linspace(vol_min,  vol_max,  vol_steps)

call_prices = np.zeros((vol_steps, spot_steps))
put_prices  = np.zeros((vol_steps, spot_steps))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        call_prices[i, j] = BlackScholes_model(spot, K, T, r, vol, 'C')
        put_prices[i, j]  = BlackScholes_model(spot, K, T, r, vol, 'P')


# Heatmaps 
def draw_heatmap(
    ax, data, xvals, yvals, title,
    title_fs=26,          
    label_fs=22,        
    tick_fs=13,           
    annot_fs=10,          
    label_pad=12          
):
    heatmap = sns.heatmap(
        data,
        xticklabels=np.round(xvals, 1),
        yticklabels=np.round(yvals, 3),
        cmap="viridis",
        annot=True,
        fmt=".2f",
        annot_kws={"color": "black", "fontsize": annot_fs},
        linewidths=0.2,
        linecolor="gray",
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=10)
    ax.set_xlabel("Spot Price", fontsize=label_fs, fontweight="bold", labelpad=label_pad)
    ax.set_ylabel("Volatility (σ)", fontsize=label_fs, fontweight="bold", labelpad=label_pad)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for label in ax.get_xticklabels():
        label.set_rotation(0)

h1, h2 = st.columns(2)

with h1:
    fig1, ax1 = plt.subplots(figsize=(16, 12), dpi=120)
    draw_heatmap(ax1, call_prices, spot_range, vol_range, "Call Price Heatmap")
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with h2:
    fig2, ax2 = plt.subplots(figsize=(16, 12), dpi=120)
    draw_heatmap(ax2, put_prices, spot_range, vol_range, "Put Price Heatmap")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.caption(
    "Black-Scholes Model (no dividends). " \
    "Prices depend on S0, K, T, r, and σ. "
    "Heatmaps show how the option price vary across variations of Spot-price and Volatility !")


# Volatility Surfaces
st.markdown("---")
st.subheader("Volatility Surfaces (Strike K, Maturity T, σ)")
st.sidebar.subheader("Volatility Surface")
K_min = st.sidebar.number_input("K min (strike)", value=float(np.round(0.7 * S0, 2)))
K_max = st.sidebar.number_input("K max (strike)", value=float(np.round(1.3 * S0, 2)))
T_min = st.sidebar.number_input("T min (years)", value=0.15, min_value=0.01)
T_max = st.sidebar.number_input("T max (years)", value=2.00, min_value=T_min + 1e-6)
vol_band = st.sidebar.slider("Market IV band (±)", min_value=0.0, max_value=0.25, value=0.05, step=0.005)
st.info(
    f"""**NOTE:**

**Presenting two diffrents Charts for Black-Scholes Model: **

** Using constant Volatility σ = {sigma:.2%} with flat surface across strikes and maturities.**

** Presenting a market-like simulation (More realistic approach) that varies within ±{vol_band:.2%} around σ, adding skew and a mild term Structure.**

**Idea: Under the constant-volatility hypothesis, the investor may be exposed to arbitrage-style mispricing: the model can **underprice puts** and **overprice calls**. That's why practitioners rely on **stochastic-volatility models (e.g., Heston)** for a more realistic view.

**Note: Prices and heatmaps display above only use constant σ (First Scenario) . The market-like surface is for comparison, not for pricing.**
"""
)

N_K, N_T = 41, 41
K_vals = np.linspace(K_min, K_max, N_K)
T_vals = np.linspace(T_min, T_max, N_T)
K_grid, T_grid = np.meshgrid(K_vals, T_vals, indexing="xy")

#  Flat IV 
IV_flat = np.full_like(K_grid, float(sigma), dtype=float)

# Market like IV 
def market_like_iv(S, K, T, base_sigma, band, tmin, tmax):
    log_m = np.log(K / S) # log-moneyness
    skew  = -np.tanh(2.0 * log_m) # Overestimate risk + insurance effect for OTM put
    midT  = 0.5 * (tmin + tmax)
    rngT  = max(tmax - tmin, 1e-9)
    term  = np.clip((T - midT) / (rngT / 2.0), -1.0, 1.0)  # light term struc (increase with T)
    shape = 0.7 * skew + 0.3 * term
    return np.clip(base_sigma + band * shape, 1e-6, None)

IV_market = market_like_iv(S0, K_grid, T_grid, sigma, vol_band, T_min, T_max)
zmin = float(min(np.nanmin(IV_flat), np.nanmin(IV_market)))
zmax = float(max(np.nanmax(IV_flat), np.nanmax(IV_market)))

# plots
tab_flat, tab_market = st.tabs(["Black-Scholes (σ constant)", "Market-like scenario"])

with tab_flat:
    fig_bs = go.Figure(data=[
        go.Surface(
            x=K_grid, y=T_grid, z=IV_flat,
            colorscale="Viridis",
            colorbar=dict(title="IV", tickformat=".2%"),
            cmin=zmin, cmax=zmax
        )
    ])
    fig_bs.update_layout(
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="Maturity T (years)",
            zaxis_title="Implied Volatility",
            zaxis=dict(tickformat=".2%")
        ),
        height=560, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_bs, use_container_width=True)

with tab_market:
    fig_mkt = go.Figure(data=[
        go.Surface(
            x=K_grid, y=T_grid, z=IV_market,
            colorscale="Viridis",
            colorbar=dict(title="IV", tickformat=".2%"),
            cmin=zmin, cmax=zmax
        )
    ])
    fig_mkt.update_layout(
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="Maturity T (years)",
            zaxis_title="Implied Volatility",
            zaxis=dict(tickformat=".2%")
        ),
        height=560, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

st.caption(
    f"Flat surface = Black–Scholes with constant σ = {sigma:.2%}. "
    f"Market-like surface = σ ± {vol_band:.2%} with skew (higher IV on lower strikes) "
    "and term structure. This is a simulation to illustrate how markets depart from the pure BS assumption."
)
