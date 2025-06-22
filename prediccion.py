import yfinance as yf
import numpy as np
import pandas as pd
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

def predecir_proximo_cierre_con_barra(ticker, barra):
    try:
        barra.text("📥 Descargando datos históricos...")
        df = yf.download(ticker, start="1900-01-01", progress=False)
        if df.empty:
            return None, None, None, None, "❌ No se pudieron obtener datos.", None

        data = df[['Close']].copy()
        data.dropna(inplace=True)

        barra.text("📊 Normalizando datos...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            ventana = data_scaled[i-sequence_length:i]
            if ventana.shape == (sequence_length, 1):
                X.append(ventana)
                y.append(data_scaled[i])
            else:
                continue

        X = np.array(X)
        y = np.array(y)

        carpeta_modelos = os.path.join(os.path.expanduser("~"), "modelos_IA_predicciones")
        os.makedirs(carpeta_modelos, exist_ok=True)
        modelo_path = os.path.join(carpeta_modelos, f"modelo_{ticker}.h5")

        barra.text("🧠 Construyendo o cargando modelo...")
        if os.path.exists(modelo_path):
            model = load_model(modelo_path)
        else:
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            barra.text("🔁 Entrenando el modelo...")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            model.save(modelo_path)

        barra.text("🔮 Prediciendo próxima vela...")
        ultimo_cierre_real = float(data['Close'].iloc[-1])
        input_data = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_price = model.predict(input_data, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)
        prediccion = predicted_price[0][0]

        señal = "📈 POSIBLE SUBIDA" if prediccion > ultimo_cierre_real else "📉 POSIBLE BAJADA"

        fecha_proxima = df.index[-1] + pd.tseries.offsets.BDay(1)
        if 'USD' in ticker or '-' in ticker:
            fecha_proxima = df.index[-1] + datetime.timedelta(days=1)

        # Explicación del patrón histórico
        ultimos = data[-5:]['Close'].values
        patron_actual = np.sign(np.diff(ultimos))

        historial = data['Close'].values
        ocurrencias = 0
        cambios_despues = []

        for i in range(60, len(historial) - 6):
            patron_pasado = np.sign(np.diff(historial[i-5:i]))
            if np.array_equal(patron_pasado, patron_actual):
                ocurrencias += 1
                cambio = ((historial[i+1] - historial[i]) / historial[i]) * 100
                cambios_despues.append(cambio)

        if ocurrencias > 0:
            media_cambio = np.mean(cambios_despues)
            if (media_cambio > 0 and prediccion > ultimo_cierre_real) or (media_cambio < 0 and prediccion < ultimo_cierre_real):
                explicacion = (
                    f"🔎 Patrón detectado: los últimos 5 días han seguido una secuencia que ya ha ocurrido {ocurrencias} veces.\n"
                    f"📊 En esos casos, el precio cambió en promedio un {media_cambio:.2f}% al día siguiente.\n"
                    f"🧠 Esto refuerza la predicción de una posible {'subida' if media_cambio > 0 else 'bajada'}."
                )
            else:
                explicacion = (
                    f"🔎 Patrón detectado: los últimos 5 días han seguido una secuencia que ya ha ocurrido {ocurrencias} veces.\n"
                    f"📊 En esos casos, el precio cambió en promedio un {media_cambio:.2f}% al día siguiente.\n"
                    f"⚠️ El comportamiento histórico contradice la predicción actual, así que se recomienda cautela."
                )
        else:
            explicacion = "🔍 Patrón no encontrado en el histórico. La predicción se basa en datos recientes y aprendizaje automático."

        # --- Gráfico Matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index[-5:], ultimos, marker='o', linestyle='-', color='orange', label='Patrón histórico (últimos 5 días)')
        ax.plot([data.index[-1], fecha_proxima], [ultimo_cierre_real, prediccion], marker='*', linestyle='--', color='red', label='Proyección próxima vela')
        ax.set_title(f"Proyección próxima vela para {ticker}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio de cierre")
        ax.legend()
        ax.grid(True)

        # Guardar en buffer en formato PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria

        return ultimo_cierre_real, prediccion, señal, fecha_proxima, explicacion, buf

    except Exception as e:
        import traceback
        error_msg = f"❌ Error interno: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, None, None, None, error_msg, None
