# prediccion.py
import yfinance as yf
import numpy as np
import pandas as pd
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

def predecir_proximo_cierre_con_barra(ticker, barra):
    try:
        barra.text("📥 Descargando datos históricos...")
        df = yf.download(ticker, start="1900-01-01", progress=False)
        if df.empty:
            return None

        data = df[['Close']].copy()
        data.dropna(inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])
            y.append(data_scaled[i])

        X, y = np.array(X), np.array(y)

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
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            model.save(modelo_path)

        ultimo_cierre_real = float(data['Close'].iloc[-1])

        input_data = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_price = model.predict(input_data, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)

        señal = "POSIBLE SUBIDA" if predicted_price[0][0] > ultimo_cierre_real else "POSIBLE BAJADA"

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
            explicacion = (
                f"🔎 En el pasado, este patrón de los últimos 5 días ocurrió {ocurrencias} veces.\n"
                f"📈 Después, el precio cambió en promedio un {media_cambio:.2f}%. "
                f"Esto respalda la predicción actual de {'subida' if media_cambio > 0 else 'bajada'}."
            )
        else:
            explicacion = "🔍 No se encontró este patrón exacto en el histórico. La predicción es puramente numérica."

        return ultimo_cierre_real, predicted_price[0][0], señal, fecha_proxima, explicacion

    except Exception as e:
        import traceback
        error_msg = f"❌ Error interno: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, None, None, None, error_msg


        return None, None, None, None, f"❌ Error interno: {str(e)}"