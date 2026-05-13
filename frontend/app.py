"""Tablero Streamlit del Proyecto Integrador de Riesgo.

Consume el backend FastAPI (no calcula). Una pestania por capa/modulo de la rubrica.
Levantar con:  streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from datetime import date

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    last_rsi = next((x for x in reversed(data["rsi_14"]) if x is not None), None)
    last_close = data["close"][-1]
    last_bbu = next((x for x in reversed(data["bb_upper"]) if x is not None), None)
    if last_rsi is not None and last_bbu is not None:
        st.info(
            f"**Interpretacion** — Cierre actual de {sel_ticker}: {last_close:.2f}. "
            f"RSI={last_rsi:.1f}: "
            f"{'zona sobrecompra (>70), posible correccion' if last_rsi > 70 else 'zona sobreventa (<30), posible rebote' if last_rsi < 30 else 'zona neutra (30-70), sin senal extrema'}. "
            f"Cierre {'por encima' if last_close > last_bbu else 'dentro o por debajo'} de la banda superior de Bollinger ({last_bbu:.2f}); "
            f"la combinacion con cruces MACD validados es el motor de las reglas de `/alertas`."
        )


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
    stats = r["stats"]
    jb_p = stats.get("jarque_bera_pvalue", 1.0)
    skew = stats.get("skew", 0.0)
    kurt = stats.get("kurtosis", 0.0)
    st.info(
        f"**Interpretacion** — Asimetria {skew:+.2f} (negativa = mas perdidas extremas que ganancias), "
        f"curtosis exceso {kurt:+.2f} (>0 = colas pesadas vs Normal). "
        f"Jarque-Bera p-valor = {jb_p:.4f}: "
        f"{'rechazamos normalidad (colas no Gaussianas)' if jb_p < 0.05 else 'no rechazamos normalidad a 5 %'}. "
        "Esto justifica usar log-rendimientos y, ante colas pesadas, VaR historico o Montecarlo "
        "en vez del parametrico Normal."
    )


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
    ewma_last = v["ewma_sigma"][-1] if v["ewma_sigma"] else None
    if ewma_last is not None:
        st.info(
            f"**Interpretacion** — Sigma EWMA actual = {ewma_last:.4f} (volatilidad diaria implicita ~ "
            f"{ewma_last*100:.2f} %, anualizada ~ {ewma_last*(252**0.5)*100:.1f} %). "
            f"EWMA es reactivo (lambda={lam}); GARCH agrega persistencia (efecto cluster). "
            f"Si el mejor por AIC es `EGARCH` o `GJR`, hay evidencia de **leverage effect** "
            "(las caidas elevan la volatilidad mas que las subidas equivalentes)."
        )


# 4. CAPM
with tabs[3]:
    st.subheader("CAPM, Beta y benchmark")
    try:
        cap = _get("/capm")
        st.write(f"**Rf diaria (FRED):** {cap['rf']:.6f} | **Benchmark:** {cap['benchmark']}")
        st.dataframe(pd.DataFrame(cap["results"]))
        if cap.get("results"):
            agresivos = [r for r in cap["results"] if r["beta"] > 1]
            defensivos = [r for r in cap["results"] if r["beta"] < 1]
            st.info(
                f"**Interpretacion** — Rf anualizada desde FRED (DGS3MO) ~ "
                f"{((1+cap['rf'])**252 - 1)*100:.2f} %. "
                f"Activos con beta>1 ({len(agresivos)}) amplifican movimientos del mercado (agresivos); "
                f"beta<1 ({len(defensivos)}) los amortiguan (defensivos). "
                "El alpha de Jensen positivo indica retornos por encima de lo que el riesgo sistematico "
                "explicaria; si es persistente, sugiere generacion de valor."
            )
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
            methods = res["methods"]
            pass_methods = [m["method"] for m in methods if m["kupiec_pass"]]
            best_method = max(methods, key=lambda m: m["kupiec_pvalue"])["method"]
            st.info(
                f"**Interpretacion** — Bajo confianza {confidence:.0%}, el portafolio puede perder "
                f"como minimo el VaR diario reportado en el {(1-confidence)*100:.0f} % peor de los dias. "
                f"CVaR (Expected Shortfall) responde 'cuanto en promedio cuando excedemos el VaR'. "
                f"Kupiec: {'metodos que pasan: ' + ', '.join(pass_methods) if pass_methods else 'ningun metodo pasa'}. "
                f"Mejor calibrado por p-valor: **{best_method}**. "
                "Si solo pasa el historico/Montecarlo, los retornos no son Normales."
            )


# 6. Markowitz
with tabs[5]:
    st.subheader("Frontera eficiente (QP) + nube Monte Carlo")
    cols = st.multiselect("Activos del portafolio", tickers, default=tickers[:4], key="mkw")
    nn = st.checkbox("Restriccion de no-negatividad", value=True)
    n_random = st.slider("Portafolios aleatorios (Monte Carlo)", 0, 10000, 5000, step=500)
    if cols and st.button("Calcular frontera"):
        with st.spinner("Resolviendo QP + simulando portafolios..."):
            res = _post(
                "/frontera-eficiente",
                {
                    "tickers": cols,
                    "non_negative": nn,
                    "n_points": 30,
                    "n_random": n_random,
                },
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        if res.get("simulated"):
            sims = pd.DataFrame(res["simulated"])
            sc = ax.scatter(
                sims["vol"], sims["ret"], c=sims["sharpe"],
                cmap="viridis", alpha=0.25, s=8, label=f"Nube ({len(sims)})",
            )
            fig.colorbar(sc, ax=ax, label="Sharpe")
        pts = pd.DataFrame([{"vol": p["vol"], "ret": p["ret"]} for p in res["points"]])
        ax.plot(pts["vol"], pts["ret"], color="red", linewidth=2, label="Frontera eficiente")
        mv = res["min_var"]
        ms = res["max_sharpe"]
        ax.scatter([mv["vol"]], [mv["ret"]], color="blue", marker="*", s=200, label="Min Var", zorder=5)
        ax.scatter([ms["vol"]], [ms["ret"]], color="orange", marker="*", s=200, label="Max Sharpe", zorder=5)
        ax.set_xlabel("Volatilidad")
        ax.set_ylabel("Retorno esperado")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        c1, c2 = st.columns(2)
        c1.metric("Min Var - Retorno", f"{mv['ret']:.4f}")
        c1.metric("Min Var - Vol", f"{mv['vol']:.4f}")
        c2.metric("Max Sharpe", f"{ms['sharpe']:.3f}")
        st.write("**Pesos Min Var:**")
        st.json(mv["weights"])
        st.info(
            f"**Interpretacion** — La curva roja es la **frontera eficiente**: combinaciones optimas "
            f"de los {len(cols)} activos en el sentido de Markowitz. La nube ({n_random} portafolios "
            "aleatorios via Dirichlet sobre el simplex) muestra todo el espacio factible; nada queda a la "
            f"izquierda de la frontera. **Min Var** ({mv['vol']:.4f}, {mv['ret']:.4f}) minimiza volatilidad sin "
            f"restriccion de retorno objetivo; **Max Sharpe** ({ms['vol']:.4f}, {ms['ret']:.4f}) "
            "maximiza retorno excedente por unidad de riesgo. Sin no-negatividad, la frontera se desplaza "
            "y los pesos pueden ser cortos (negativos)."
        )


# 7. Senales
with tabs[6]:
    st.subheader("Senales activas")
    s = _get("/alertas")
    if s["signals"]:
        st.dataframe(pd.DataFrame(s["signals"]))
        buys = sum(1 for x in s["signals"] if x.get("side") == "buy")
        sells = sum(1 for x in s["signals"] if x.get("side") == "sell")
        st.info(
            f"**Interpretacion** — Detectadas {len(s['signals'])} senales hoy "
            f"({buys} de compra, {sells} de venta). Reglas: RSI sobreventa/sobrecompra, "
            "cruces de MACD vs signal y ruptura de bandas de Bollinger. Cada deteccion se "
            "persiste en `signals_log` para auditoria y backtesting posterior. "
            "Ajusta los umbrales en la URL del endpoint si quieres reglas mas estrictas."
        )
    else:
        st.info("Sin senales activas en este momento.")


# 8. Macro
with tabs[7]:
    st.subheader("Macro - tasa libre de riesgo")
    m = _get("/macro")
    c1, c2 = st.columns(2)
    c1.metric("Rf anual", f"{m['rf']*100:.2f}%")
    c2.write(f"Fuente: `{m['rf_source']}`")
    st.info(
        f"**Interpretacion** — La tasa libre de riesgo es la referencia que separa la "
        f"compensacion por tiempo (Rf) de la compensacion por asumir riesgo (premio de mercado). "
        f"Se usa como entrada en CAPM, Black-Scholes y descuento de cupones del bono. "
        f"Cache TTL 24 h: una caida de FRED no rompe la API gracias al fallback estatico. "
        f"Fuente '{m['rf_source']}' indica si vino de FRED o del fallback."
    )


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
    st.info(
        f"**Interpretacion de NS** — beta0={c['ns_beta0']:.3f} es el nivel asintotico (tasas a muy largo plazo). "
        f"beta1={c['ns_beta1']:.3f}: la pendiente; si negativa, la curva sube con la madurez (expansion). "
        f"beta2={c['ns_beta2']:.3f}: la curvatura/joroba; tau={c['ns_tau']:.2f} fija donde aparece. "
        f"Curva invertida (beta0+beta1<beta0) suele anticipar recesion."
    )

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
        st.info(
            f"**Interpretacion** — Precio={b['price']:.2f}, "
            f"D modificada={b['modified_duration']:.3f}: una subida de tasas de 1 % bajara el precio "
            f"aproximadamente {b['modified_duration']*100:.2f} %. Convexidad={b['convexity']:.3f} corrige "
            "esa estimacion lineal (curva 'D+C' siempre acerca al reprice exacto). "
            "Para shocks grandes (>=100 bp), la diferencia entre `linear_D` y `exact` revela la "
            "no-linealidad del precio."
        )


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
        g = o["greeks"]
        st.info(
            f"**Interpretacion** — Delta={g['delta']:+.3f}: cuanto cambia el precio de la opcion "
            f"ante un movimiento unitario del subyacente (sirve para hedge); "
            f"Gamma={g['gamma']:.4f}: la convexidad del delta. "
            f"Vega={g['vega']:.3f}: sensibilidad a volatilidad implicita (1 %); "
            f"Theta={g['theta']:+.3f}: erosion diaria del precio (time decay). "
            f"Paridad call-put ≈ 0 ({o['parity_check']:.2e}) confirma consistencia interna del modelo "
            "y permite implied vol via Newton-Raphson."
        )


# 11. Stress
with tabs[10]:
    st.subheader("Stress testing - 5 escenarios spec CIII")
    cols = st.multiselect("Activos", tickers, default=tickers[:3], key="strs")
    if cols and st.button("Correr escenarios"):
        weights = {t: round(1.0 / len(cols), 4) for t in cols}
        weights[cols[-1]] = round(1.0 - sum(list(weights.values())[:-1]), 4)
        s = _post("/stress", {"weights": weights})
        st.metric("VaR base (historico)", f"{s['base_var']:.4f}")

        sc_df = pd.DataFrame(s["scenarios"]).set_index("name")

        # (a) Bar chart de perdidas por escenario.
        st.markdown("**Perdida anualizada del portafolio por escenario**")
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        sc_df["portfolio_loss"].plot(
            kind="bar", color="firebrick", ax=ax1, edgecolor="black"
        )
        ax1.set_ylabel("Perdida (fraccion)")
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=20)
        st.pyplot(fig1)
        st.caption(
            "El escenario `combined` integra los 3 shocks; tipicamente es el peor."
        )

        # (b) Comparacion VaR base vs VaR estresado.
        st.markdown("**VaR base vs VaR estresado**")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        sc_df[["var_base", "var_stressed"]].plot(
            kind="bar", ax=ax2, edgecolor="black",
            color=["steelblue", "darkorange"],
        )
        ax2.set_ylabel("VaR (95 %)")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.tick_params(axis="x", rotation=20)
        ax2.legend(["Base", "Estresado"])
        st.pyplot(fig2)
        st.caption(
            "Bajo cualquier escenario adverso el VaR estresado debe ser mayor que el base."
        )

        # (c) Heatmap activo x escenario (sensibilidad).
        st.markdown("**Sensibilidad por activo (contribucion al impacto)**")
        sens_df = pd.DataFrame(
            {sc["name"]: sc["sensitivity_by_asset"] for sc in s["scenarios"]}
        )
        fig3, ax3 = plt.subplots(figsize=(8, max(3, 0.4 * len(sens_df) + 2)))
        sns.heatmap(
            sens_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            cbar_kws={"label": "Contribucion"}, ax=ax3,
        )
        ax3.set_xlabel("Escenario")
        ax3.set_ylabel("Activo")
        st.pyplot(fig3)
        st.caption(
            "Valores rojos: el activo amplifica el shock; azules: lo amortigua."
        )

        with st.expander("Tabla completa"):
            st.dataframe(sc_df)
        peor = sc_df["portfolio_loss"].idxmin()
        st.info(
            f"**Interpretacion** — El escenario mas adverso es `{peor}` con perdida estimada de "
            f"{sc_df.loc[peor, 'portfolio_loss']*100:+.2f} % anual. "
            "La columna `sensitivity_by_asset` indica que activos amplifican el shock vs cuales lo "
            "amortiguan, util para rebalancear el portafolio antes de eventos previstos. "
            "Spec CIII exige 4 escenarios obligatorios + combinado; aqui implementamos los 5."
        )


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
            st.info(
                f"**Interpretacion** — El modelo es un GradientBoostingClassifier sobre {len(p['features_used'])} "
                f"features tecnicas (lags de retornos, EWMA, RSI, MACD, ratio EMA). El patron Singleton garantiza "
                "que el `.joblib` se carga UNA vez al arrancar la app (verificable en logs del backend) y "
                f"cada inferencia toma <100 ms. Esta llamada se registro en `prediction_log` con id={p.get('log_id', 'n/a')}; "
                "puedes hacer back-fill del valor real con `POST /predict/{log_id}/actual` "
                "para habilitar monitoreo de drift."
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
