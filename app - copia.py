# app.py
import streamlit as st
from prediccion import predecir_proximo_cierre_con_barra

st.set_page_config(page_title="Predicción Bursátil AI", layout="centered")
st.title("📈 Predicción de próxima vela con IA")

with st.form("form_prediccion"):
    ticker = st.text_input("Introduce el símbolo de la acción o criptomoneda (ej: AAPL, BTC-USD):")
    submitted = st.form_submit_button("Predecir")

if submitted and ticker:
    with st.spinner("Cargando modelo y analizando datos..."):
        barra = st.empty()
        resultado = predecir_proximo_cierre_con_barra(ticker.upper(), barra)

    if resultado and all(r is not None for r in resultado[:4]):
        precio_real, prediccion, señal, fecha, explicacion = resultado
        st.markdown(f"### 🔮 Predicción próxima vela ({fecha.strftime('%A %d de %B de %Y')}): {prediccion:.2f} USD")
        st.markdown(f"### 📢 Señal: **{señal}**")
        st.markdown(f"### 💰 Último cierre real: {precio_real:.2f} USD")
        st.markdown("---")
        st.markdown(f"🧠 **Explicación de la IA:**\n\n{explicacion}")
    else:
        mensaje_error = resultado[-1] if resultado and resultado[-1] else "❌ No se pudo obtener la predicción."
        st.error(mensaje_error)
