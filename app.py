import streamlit as st
from prediccion import predecir_proximo_cierre_con_barra
from PIL import Image
import io

st.set_page_config(page_title="Predicción Bursátil AI", layout="centered")
st.title("📈 Predicción de próxima vela con IA")

with st.form("form_prediccion"):
    ticker = st.text_input("Introduce el símbolo de la acción o criptomoneda (ej: AAPL, BTC-USD):")
    submitted = st.form_submit_button("Predecir")

if submitted and ticker:
    with st.spinner("Cargando modelo y analizando datos..."):
        barra = st.empty()
        resultado = predecir_proximo_cierre_con_barra(ticker.upper(), barra)

    if resultado and all(r is not None for r in resultado[:5]):
        precio_real, prediccion, señal, fecha, explicacion, buf_img = resultado
        st.markdown(f"### 🔮 Predicción próxima vela ({fecha.strftime('%A %d de %B de %Y')}): {prediccion:.2f} USD")
        st.markdown(f"### 📢 Señal: **{señal}**")
        st.markdown(f"### 💰 Último cierre real: {precio_real:.2f} USD")
        st.markdown("---")
        st.markdown(f"🧠 **Explicación de la IA:**\n\n{explicacion}")

        # Mostrar imagen PNG de la gráfica
        image = Image.open(buf_img)
        st.image(image, use_container_width=True)

    else:
        mensaje_error = resultado[-2] if resultado and resultado[-2] else "❌ No se pudo obtener la predicción."
        st.error(mensaje_error)
