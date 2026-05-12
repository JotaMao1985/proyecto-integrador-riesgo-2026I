"""Tablero Streamlit del Proyecto Integrador de Riesgo.

Consume el backend FastAPI (no calcula). Una pestania por capa/modulo de la rubrica.
Levantar con:  streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from datetime import date

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
TIMEOUT_S = 30.0


st.set_page_config(
    page_title="Riesgo USTA - Tablero",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(ttl=300, show_spinner=False)
def _get(path: str, params: dict | None = None) -> dict | list:
    r = httpx.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _post(path: str, payload: dict) -> dict | list:
    r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


# ---------- Header ----------
st.title("Tablero de Riesgo - USTA 2026-I")
st.caption(
    "Solucion de referencia. Backend: "
    f"`{API_BASE}` - "
    "[/docs]({API_BASE}/docs)  - "
    "Cinco capas: datos, analisis clasico, renta fija/derivados, ML, deploy."
)

try:
    health = _get("/health")
    st.success(f"Backend OK (env={health['env']}, app={health['app_name']})")
except Exception as exc:
    st.error(f"Backend no responde en {API_BASE}: {exc}")
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Activos")
    activos = _get("/activos")
    tickers = [a["ticker"] for a in activos]
    st.write(pd.DataFrame(activos))
    sel_ticker = st.selectbox("Ticker activo", tickers, index=0)


# ---------- Tabs ----------
tabs = st.tabs(
    [
        "1. Tecnico",
        "2. Rendimientos",
        "3. Volatilidad",
        "4. CAPM",
        "5. VaR",
        "6. Markowitz",
        "7. Senales",
        "8. Macro",
        "9. Renta fija",
        "10. Opciones",
        "11. Stress",
        "12. ML",
    ]
)


# 1. Tecnico
with tabs[0]:
    st.subheader("Indicadores tecnicos")
    data = _get(f"/indicadores/{sel_ticker}")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates"]),
            "close": data["close"],
            "sma_20": data["sma_20"],
            "ema_20": data["ema_20"],
            "bb_upper": data["bb_upper"],
            "bb_lower": data["bb_lower"],
        }
    ).set_index("date")
    st.line_chart(df)
    c1, c2 = st.columns(2)
    with c1:
        rsi_df = pd.DataFrame({"date": pd.to_datetime(data["dates"]), "rsi_14": data["rsi_14"]}).set_index("date")
        st.line_chart(rsi_df)
        st.caption("RSI 14. > 70 sobrecompra, < 30 sobreventa.")
    with c2:
        macd_df = pd.DataFrame(
            {
                "date": pd.to_datetime(data["dates"]),
                "macd": data["macd"],
                "macd_signal": data["macd_signal"],
            }
        ).set_index("date")
        st.line_chart(macd_df)
        st.caption("MACD vs senial.")


# 2. Rendimientos
with tabs[1]:
    st.subheader("Rendimientos y propiedades empiricas")
    r = _get(f"/rendimientos/{sel_ticker}")
    df = pd.DataFrame(
        {"date": pd.to_datetime(r["dates"]), "log_return": r["log"]}
    ).set_index("date")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df)
        st.caption("Log-rendimientos diarios.")
    with c2:
        st.write("**Estadisticas descriptivas:**")
        st.json(r["stats"])


# 3. Volatilidad
with tabs[2]:
    st.subheader("Volatilidad EWMA + GARCH")
    lam = st.slider("Lambda EWMA", 0.51, 0.99, 0.94, step=0.01)
    with st.spinner("Ajustando GARCH..."):
        v = _get(f"/volatilidad/{sel_ticker}", params={"ewma_lambda": lam})
    df = pd.DataFrame(
        {"date": pd.to_datetime(v["dates"]), "ewma_sigma": v["ewma_sigma"]}
    ).set_index("date")
    st.line_chart(df)
    st.write(f"**Mejor modelo GARCH por AIC:** `{v['best_model']}`")
    st.dataframe(pd.DataFrame(v["garch_results"]))


# 4. CAPM
with tabs[3]:
    st.subheader("CAPM, Beta y benchmark")
    try:
        cap = _get("/capm")
        st.write(f"**Rf diaria (FRED):** {cap['rf']:.6f} | **Benchmark:** {cap['benchmark']}")
        st.dataframe(pd.DataFrame(cap["results"]))
    except Exception as exc:
        st.warning(f"Necesita datos de SPY en BD. {exc}")


# 5. VaR
with tabs[4]:
    st.subheader("VaR + CVaR + Kupiec")
    cols = st.multiselect("Activos en el portafolio", tickers, default=tickers[:3])
    if cols:
        weights = {t: round(1.0 / len(cols), 4) for t in cols}
        # Ajustar para que sumen exacto 1.
        weights[cols[-1]] = round(1.0 - sum(list(weights.values())[:-1]), 4)
        confidence = st.slider("Nivel de confianza", 0.90, 0.99, 0.95, step=0.01)
        if st.button("Calcular VaR"):
            res = _post("/var", {"weights": weights, "confidence": confidence, "n_simulations": 10000})
            st.dataframe(pd.DataFrame(res["methods"]))
            st.caption("Kupiec: pasa si p-valor > 0.05.")


# 6. Markowitz
with tabs[5]:
    st.subheader("Frontera eficiente (QP)")
    cols = st.multiselect("Activos del portafolio", tickers, default=tickers[:4], key="mkw")
    nn = st.checkbox("Restriccion de no-negatividad", value=True)
    if cols and st.button("Calcular frontera"):
        with st.spinner("Resolviendo QP..."):
            res = _post(
                "/frontera-eficiente",
                {"tickers": cols, "non_negative": nn, "n_points": 30},
            )
        df = pd.DataFrame([{"vol": p["vol"], "ret": p["ret"]} for p in res["points"]])
        st.scatter_chart(df, x="vol", y="ret")
        c1, c2 = st.columns(2)
        c1.metric("Min Var - Retorno", f"{res['min_var']['ret']:.4f}")
        c1.metric("Min Var - Vol", f"{res['min_var']['vol']:.4f}")
        c2.metric("Max Sharpe", f"{res['max_sharpe']['sharpe']:.3f}")
        st.write("**Pesos Min Var:**")
        st.json(res["min_var"]["weights"])


# 7. Senales
with tabs[6]:
    st.subheader("Senales activas")
    s = _get("/alertas")
    if s["signals"]:
        st.dataframe(pd.DataFrame(s["signals"]))
    else:
        st.info("Sin senales activas en este momento.")


# 8. Macro
with tabs[7]:
    st.subheader("Macro - tasa libre de riesgo")
    m = _get("/macro")
    c1, c2 = st.columns(2)
    c1.metric("Rf anual", f"{m['rf']*100:.2f}%")
    c2.write(f"Fuente: `{m['rf_source']}`")


# 9. Renta fija
with tabs[8]:
    st.subheader("Curva de rendimiento + Nelson-Siegel")
    c = _get("/curva-rendimiento")
    df = pd.DataFrame(
        {"maturity": c["maturities"], "observed": c["yields"], "ns_fitted": c["fitted"]}
    ).set_index("maturity")
    st.line_chart(df)
    st.write(f"**RMSE del ajuste NS:** {c['rmse']:.6f}")
    st.write(f"**Parametros:** beta0={c['ns_beta0']:.4f}, beta1={c['ns_beta1']:.4f}, beta2={c['ns_beta2']:.4f}, tau={c['ns_tau']:.4f}")

    st.divider()
    st.subheader("Bono sintetico - duracion y convexidad")
    c1, c2 = st.columns(2)
    with c1:
        coupon = st.number_input("Cupon", 0.0, 0.20, 0.05, step=0.005)
        ytm = st.number_input("YTM", 0.001, 0.20, 0.05, step=0.005)
    with c2:
        years = st.number_input("Anios", 0.5, 30.0, 10.0, step=0.5)
        cpy = st.selectbox("Cupones por anio", [1, 2, 4], index=1)
    if st.button("Valorar bono"):
        b = _post(
            "/bono/duracion",
            {
                "face_value": 1000,
                "coupon_rate": coupon,
                "ytm": ytm,
                "years": years,
                "coupons_per_year": int(cpy),
            },
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"{b['price']:.2f}")
        m2.metric("D Macaulay", f"{b['macaulay_duration']:.3f}")
        m3.metric("D modificada", f"{b['modified_duration']:.3f}")
        m4.metric("Convexidad", f"{b['convexity']:.3f}")
        st.write("**Sensibilidad por shock:**")
        st.dataframe(pd.DataFrame(b["sensitivity"]).T)


# 10. Opciones
with tabs[9]:
    st.subheader("Black-Scholes + Greeks")
    c1, c2, c3 = st.columns(3)
    with c1:
        spot = st.number_input("Spot", 1.0, 1000.0, 100.0)
        strike = st.number_input("Strike", 1.0, 1000.0, 100.0)
    with c2:
        t = st.number_input("T (anios)", 0.05, 5.0, 1.0)
        rf = st.number_input("Rf", 0.0, 0.20, 0.05, step=0.005)
    with c3:
        sigma = st.number_input("Sigma", 0.01, 2.0, 0.20, step=0.01)
        opt = st.selectbox("Tipo", ["call", "put"])
    if st.button("Valorar opcion"):
        o = _post(
            "/opcion/precio",
            {
                "spot": spot,
                "strike": strike,
                "time_to_expiry": t,
                "rf": rf,
                "sigma": sigma,
                "option_type": opt,
            },
        )
        st.metric("Precio", f"{o['price']:.4f}")
        st.write("**Greeks:**")
        st.json(o["greeks"])
        st.caption(f"Verificacion put-call parity: {o['parity_check']:.2e}")


# 11. Stress
with tabs[10]:
    st.subheader("Stress testing")
    cols = st.multiselect("Activos", tickers, default=tickers[:3], key="strs")
    if cols and st.button("Correr escenarios"):
        weights = {t: round(1.0 / len(cols), 4) for t in cols}
        weights[cols[-1]] = round(1.0 - sum(list(weights.values())[:-1]), 4)
        s = _post("/stress", {"weights": weights})
        st.metric("VaR base", f"{s['base_var']:.4f}")
        st.dataframe(pd.DataFrame(s["scenarios"]))


# 12. ML
with tabs[11]:
    st.subheader("Modelo ML - direccion next-day")
    st.caption(
        "Si el endpoint falla con 500, entrenar primero: "
        "`python -m app.ml.train` dentro de backend/."
    )
    if st.button("Predecir"):
        try:
            p = _post("/predict", {"ticker": sel_ticker, "lookback_days": 120})
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediccion", "Sube" if p["prediction"] == 1 else "Baja")
            c2.metric("Probabilidad", f"{p['probability']:.2%}")
            c3.metric("Version", p["model_version"])
            st.write("**Features usadas:**", p["features_used"])
        except Exception as exc:
            st.error(f"Error: {exc}")
