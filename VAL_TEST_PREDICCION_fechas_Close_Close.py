#ITERA FECHA POR FECHA Y VA A HACIENDO UNA PREDICCIÓN DE CADA FECHA CON CRITERIO DE ENTRADA EL CIERRE DEL DÍA DE LA PREDICCIÓN Y SALIDA OTRO CIERRE

#UTILIZA LA MISMA ARQUITECTURA QUE EN LOS DE VAL_TEST, DONDE UNA VEZ QUE TENGA LOS MEJORES PARÁMETROS, YA LO PASO AQUÍ Y HAGO LA PREDICCIÓN FECHA POR FECHA SIN DIVIDIR EN TEST NI VALIDACIÓN

import numpy as np
import pandas as pd
import os
import yfinance as yf
import tensorflow as tf
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense # Keras's layers

from keras.layers import Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed

from datetime import datetime
from datetime import timedelta

from tensorflow import keras
tf.random.set_seed(0)

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suprimir solo ese warning
warnings.simplefilter("ignore", ValueWarning)



from matplotlib import pyplot as plt

import yfinance as yf, pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

from stockstats import StockDataFrame as Sdf

#from tti.indicators import AccumulationDistributionLine
from tti.indicators import *

#PARA df4
""" import talib
from talib import MA_Type
from talib import abstract
from talib.abstract import * """

#PARA df5
from finta import TA

#PARA df6
import pandas_ta as ta

import time
from datetime import date
import time
#Esto lo comento, solo es para que en los cuadernos Jupyter no de warnings
#%matplotlib inline

# download the data
#df = yf.download(tickers=['AAPL'], period='1y')
stock = 'UNH'
sector = "XLV"
cap = 500
#tipo_activo = 'ETF'
tipo_activo = 'STOCK'

arquitectura = "ARIMA"
arquitectura = "SARIMAX"             
arquitectura = "RANDOM_FOREST"
arquitectura = "XGBOOST"
arquitectura = "LIGHTGBM"
arquitectura = "GRU"    
arquitectura = "LSTM"
#arquitectura = "Conv1D_LSTM_ECA"
#arquitectura = "Atention_LSTM_TCN_Transformer_Ensemble"



start_date_prediction = "2020-01-01"

predicted_feature = "Close" 
predicted_feature = "porcentaje_cambio"
#predicted_feature = "target_class"

num_indicadores = 6
#Se pueden filtrar las columnas que quiera, si lo quiero por num_indicadores, se pone como  comentario
cols = ["Close", "High", "Low", "Open", "Volume", "RSI_14", "MACD_signal", "EMA_10", "EMA_50", "kalman", "bollinger_mid", "bollinger_upper" ,"bollinger_lower"]

# generate the input and output sequences
n_lookback = 30  # length of  input sequences (lookback period)
n_forecast = 1  # length of output sequences (forecast period)

epochs = 10
batch_size = 64

layer_size = 64
dropout = 0.05


learning_rate = 0.005

criterio_entrada_salida = "Close_Close"

#CON DIFERENCIA SI QUIERO QUITAR EL PRIMER VALOR DE LA PREDICCIÓN, YA QUE A VECES EL PRIMERO VALOR TINE UN OFFSET MUY ALTO
con_diferencia = "NO"
#PARA MOSTRAR LOS PLOT DE LAS PREDICCIONES EN CADA ITERACIÓN Y SE VAYAN ABRIENDO VENTANAS PARA VERLO
mostrar_plot = "NO"
#LO PONEMOS EN SI, SI NO QUEREMOS QUE GUARDE EL MODELO Y SOLO ENTRENE CADA DÍA, SINO QUE CADA INCREMENTO ENTRENE DESDE EL PRINCIPIO DE NUEVO
loading_model = "NO"


# === Loss Functions disponibles ===

# 🔹 Para REGRESIÓN
""" loss_function = "mean_squared_error"   # (MSE) clásico, penaliza mucho errores grandes
loss_function = "mean_absolute_error"  # (MAE) más robusto a outliers
loss_function = "huber"                # combina MSE y MAE, útil con outliers
loss_function = "log_cosh"             # versión suave del MAE, estable

# 🔹 Para CLASIFICACIÓN BINARIA
loss_function = "binary_crossentropy"  # más usado, predicciones 0/1
loss_function = "hinge"                # usado en SVMs, margen máximo
loss_function = "squared_hinge"        # variante suavizada de hinge
loss_function = "logloss"              # usado en LightGBM/XGBoost

# 🔹 Para CLASIFICACIÓN MULTICLASE
loss_function = "categorical_crossentropy"  # con one-hot encoding
loss_function = "sparse_categorical_crossentropy"  # con etiquetas enteras """

loss_function = "mean_squared_error"

# Ajuste automático de loss_function
if predicted_feature == "target_class" and arquitectura in ["RANDOM_FOREST", "XGBOOST", "LIGHTGBM"]:
    # Clasificación binaria
    if arquitectura == "RANDOM_FOREST":
        loss_function = "log_loss"       # sklearn
    elif arquitectura == "XGBOOST":
        loss_function = "logloss"        # xgboost
    elif arquitectura == "LIGHTGBM":
        loss_function = "binary_logloss" # lightgbm
else:
    # Regresión
    if arquitectura == "RANDOM_FOREST":
        # sklearn acepta 'squared_error', 'absolute_error', 'poisson', 'gamma'
        if loss_function == "mean_squared_error":
            loss_function = "squared_error"
    elif arquitectura == "XGBOOST":
        # xgboost acepta 'rmse', 'mae', 'logloss', 'mlogloss'
        if loss_function == "mean_squared_error":
            loss_function = "rmse"
    elif arquitectura == "LIGHTGBM":
        # lightgbm acepta 'l2', 'mae', 'rmse', 'binary_logloss'
        if loss_function == "mean_squared_error":
            loss_function = "l2"
    
# === Optimizers disponibles ===

""" optimizer = "adam"      # estándar, rápido y estable, buen punto de partida
optimizer = "sgd"       # Stochastic Gradient Descent, clásico pero más lento
optimizer = "rmsprop"   # bueno para RNN/LSTM, ajusta learning rate por dimensión
optimizer = "adagrad"   # útil si hay features poco frecuentes
optimizer = "adadelta"  # variante adaptativa de adagrad
optimizer = "nadam"     # Adam + Nesterov momentum, suele converger más rápido
optimizer = "ftrl"      # especializado en modelos dispersos y online learning """

optimizer = "adam"

# Si existe cols y no existe la variable a predecir, la tiene que añadir
if 'cols' in locals() or 'cols' in globals():
    if predicted_feature not in cols:
        cols.append(predicted_feature)


#Si solo predigo 1 valor, como hago Shift(1), no tiene sentido poner con_diferencia = "SI"
if (n_forecast == 1):
    con_diferencia = "NO"


    
#SI PREDECIMOS EL PORCENTAJE DE CAMBIO, VA A SER EL DIA SIGUIENTE
if (predicted_feature == "porcentaje_cambio"):
    n_forecast = 1
    #No puede tener menos de 6 features porque sino no pilla el porcentaje_cambio
    if (num_indicadores < 6):
        num_indicadores = 6

#SI PREDECIMOS TARGET CLASS, VA A SER EL DIA SIGUIENTE
if (predicted_feature == "target_class"):
    n_forecast = 1
    #No puede tener menos de 7 features porque sino no pilla el porcentaje_cambio
    if (num_indicadores < 7):
        num_indicadores = 7


fecha_a_predecir_path = f"fecha_a_predecir_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.txt"
# Verifica si el archivo existe
if os.path.exists(fecha_a_predecir_path):
    with open(fecha_a_predecir_path, "r") as f:
        fecha_str = f.read().strip()  # Elimina espacios o saltos de línea
        start_date_prediction = pd.to_datetime(fecha_str)


#Obtenemos el SECTOR y el PAIS de la acción
from yahooquery import Ticker

try:
    def get_stock_info(symbol):
        stock = yf.Ticker(symbol)
        info = stock.info
        country = info['country']
        sector = info['sector']
        return country, sector

    country, industry = get_stock_info(stock)
    print(country)

    print(industry)

    if (tipo_activo == 'STOCK'):
        sector = industry
        country = country

        if "Technology" in sector:
            sector = 'XLK'
        if "Healthcare" in sector:
            sector = 'XLV'
        if "Financial" in sector:
            sector = 'XLF'
        if "Communication" in sector:
            sector = 'XLC'
        if "Consumer Cyclical" in sector:
            sector = 'XLY'
        if "Industrials" in sector:
            sector = 'XLI'
        if "Consumer Defensive" in sector:
            sector = 'XLP'
        if "Energy" in sector:
            sector = 'XLE'
        if "Utilities" in sector:
            sector = 'XLU'
        if "Real Estate" in sector:
            sector = 'XLRE'
        if "Basic Materials" in sector:
            sector = 'XLB'

        print(sector)
except:
    sector = sector


df_original = pd.read_csv(f"{stock}.csv", index_col=0, parse_dates=True)


# Ajustar num_indicadores si es mayor que el número de columnas disponibles
if num_indicadores > df_original.shape[1]:
    num_indicadores = df_original.shape[1]



if "cols" in locals():
    df_original = df_original[cols]
    num_indicadores = len(cols)
else:
    # Seleccionar las primeras 'num_indicadores' columnas
    df_original = df_original.iloc[:, :num_indicadores]






# Selección de FEATURES y TARGET
features_cols = df_original.columns[:num_indicadores].tolist()  # múltiples features
assert predicted_feature in df_original.columns, f"{predicted_feature} no está en el DataFrame"

is_classification = (predicted_feature == "target_class")



# Filtrar las fechas iguales o posteriores a esa fecha
fechas = df_original.index[df_original.index >= start_date_prediction]

import joblib

from tensorflow.keras.models import load_model

modelo_path = f"modelo_lstm_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.keras"
scaler_path = f"modelo_scaler_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.gz"


# Recorrer cada fecha
for fecha in fechas:
    # Supongamos que `fecha` es del tipo Timestamp o datetime
    fecha_str = fecha.strftime('%Y-%m-%d')
    #Escribo el fichero al principio, ya que es la fecha por la que va, y luego va a empezar por esta, si lo excribo al final, cuando lo pare y vuelva a ejecutar, va a repetir la ultima fecha
    # Escribir la fecha en el archivo (crea o sobrescribe)
    with open(fecha_a_predecir_path, 'w') as f:
        f.write(fecha_str)


    df = df_original[df_original.index <= fecha]

    longitud_datos = len(df)

    print(fecha.strftime('%Y-%m-%d'))

    end_date = df.index[-1].strftime('%Y-%m-%d')


    with open(fecha_a_predecir_path, 'w') as f:
        f.write(fecha_str)

    print(f"\n📅 Procesando fecha: {fecha_str}")

    # Eliminar NaNs
    df = df.copy().dropna()

    if "cols" in locals():
        df = df[cols]
        num_indicadores = len(cols)
    

    if arquitectura in ["ARIMA", "SARIMAX"]:
        last_loss = 0
        last_mae = 0
        last_mape = 0
        last_mse = 0
        last_msle = 0

        n_forecast = 1
        epochs = 0
        batch_size = 0
        layer_size = 0
        dropout = 0
        learning_rate = 0

        loss_function = 'NA'
        optimizer = 'NA'

        from pmdarima import auto_arima
        

        if is_classification:
            raise ValueError("ARIMA/SARIMAX no es compatible con clasificación ('target_class').")

        print(f"🔍 Entrenando modelo {arquitectura} con auto_arima...")

        # Serie completa en escala real
        y_series = df[predicted_feature]

        # Detectar si usar estacionalidad
        is_seasonal = arquitectura == "SARIMAX"

        # Entrenamiento con búsqueda automática de parámetros
        stepwise_model = auto_arima(
            y_series,
            seasonal=is_seasonal,
            m=5 if is_seasonal else 1,  # Usar 5 para días laborables, 1 si ARIMA
            trace=True,                 # ✅ Muestra AIC, parámetros, etc.
            error_action="ignore",
            suppress_warnings=False,
            stepwise=True
        )

        # Mostrar resumen del modelo
        print(stepwise_model.summary())

        # Predecir siguiente valor
        sm_model = stepwise_model.fit(y_series)
        forecast = sm_model.predict(n_periods=1).iloc[0]

        print("📈 Forecast (siguiente valor):", forecast)




    elif (arquitectura == "RANDOM_FOREST"):
        
        last_loss = 0
        last_mae = 0
        last_mape = 0
        last_mse = 0
        last_msle = 0

        n_forecast = 1
        epochs = 0
        batch_size = 0
        layer_size = 0
        dropout = 0
        learning_rate = 0

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        import joblib

        # ========= CONFIGURACIÓN =========
        if (epochs < 5):
            epochs = 5


        # ========= ESCALADO =========
        if predicted_feature not in df.columns:
            raise ValueError(f"La columna '{predicted_feature}' no está en el DataFrame")

        predicted_col_idx = df.columns.get_loc(predicted_feature)

        if is_classification:
            print("🔍 Clasificación multiclase (-1, 0, 1): no se aplica escalado.")
            scaler = None
            scaled_data = df.values
        else:
            print("🔧 Escalando datos para regresión...")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(df)
            joblib.dump(scaler, "scaler.pkl")

        # ========= CREAR SECUENCIAS =========
        X, Y = [], []
        for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
            X.append(scaled_data[i - n_lookback:i])
            Y.append(scaled_data[i:i + n_forecast, predicted_col_idx])

        X = np.array(X)
        Y = np.array(Y).reshape(-1, n_forecast)

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("❌ Error: no se pudo construir X correctamente.")

        # ========= APLANAR PARA MODELO CLÁSICO =========
        X_2d = X.reshape(X.shape[0], -1)

        # ========= Clasificación: mapear (-1 → 0, 0 → 1, 1 → 2) =========
        if is_classification and predicted_feature == "target_class":
            y_mapped = np.where(Y.flatten() == -1, 0, np.where(Y.flatten() == 0, 1, 2))
        else:
            y_mapped = Y.flatten()

        # ========= ENTRENAMIENTO =========
        print("🚀 Entrenando modelo Random Forest...")

        if is_classification:
            model = RandomForestClassifier(n_estimators=epochs, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=epochs, random_state=42)

        model.fit(X_2d, y_mapped)

        # ========= PREDICCIÓN PARA MAÑANA =========
        last_block = scaled_data[-n_lookback:].reshape(1, -1)

        if is_classification and predicted_feature == "target_class":
            pred_proba = model.predict_proba(last_block)[0]  # [p_clase_0, p_clase_1, p_clase_2]
            predicted_index = int(np.argmax(pred_proba))     # 0, 1, 2
            forecast = [-1, 0, 1][predicted_index]
        else:
            forecast_scaled = model.predict(last_block)[0]
            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, predicted_col_idx] = forecast_scaled
            forecast = scaler.inverse_transform(dummy)[0, predicted_col_idx]

        # ========= RESULTADO =========
        print("📈 Forecast para mañana:", forecast)

    elif arquitectura == "XGBOOST":
        

        n_forecast = 1
        batch_size = 0
        layer_size = 0
        dropout = 0
        porcentaje_validacion = 0
        optimizer = 'NA'

        
        
        if (loss_function == "mean_squared_error"):
            loss_function = "rmse"  # puede ser "mae", "mse", etc.


        last_loss = 0
        last_mae = 0
        last_mape = 0
        last_mse = 0
        last_msle = 0


        import numpy as np
        import joblib
        from sklearn.preprocessing import MinMaxScaler
        import xgboost as xgb
        from collections import Counter

        # ========== Hiperparámetros ==========
        

        if predicted_feature not in df.columns:
            raise ValueError(f"❌ La columna '{predicted_feature}' no está en el DataFrame")

        predicted_col_idx = df.columns.get_loc(predicted_feature)

        # ========= ESCALADO =========
        if predicted_feature not in df.columns:
            raise ValueError(f"❌ La columna '{predicted_feature}' no está en el DataFrame")

        predicted_col_idx = df.columns.get_loc(predicted_feature)

        if is_classification:
            print("🔍 Clasificación multiclase (-1, 0, 1): no se aplica escalado.")
            scaler = None
            scaled_data = df.values
        else:
            print("🔧 Escalando datos...")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(df)
            joblib.dump(scaler, "scaler.pkl")

        # ========= CREAR SECUENCIAS =========
        X, Y = [], []
        for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
            X.append(scaled_data[i - n_lookback:i])
            Y.append(scaled_data[i:i + n_forecast, predicted_col_idx])

        X = np.array(X)
        Y = np.array(Y).reshape(-1, n_forecast)

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("❌ Error: no se pudo construir X correctamente.")

        X_2d = X.reshape(X.shape[0], -1)

        # ========= MAPEO DE CLASES: (-1 → 0, 0 → 1, 1 → 2) =========
        if is_classification and predicted_feature == "target_class":
            y_mapped = np.where(Y.flatten() == -1, 0, np.where(Y.flatten() == 0, 1, 2))
        else:
            y_mapped = Y.flatten()

        # ========= CREAR DMATRIX =========
        dtrain = xgb.DMatrix(X_2d, label=y_mapped)

        # ========= PARÁMETROS XGBOOST =========
        xgb_params = {
            "verbosity": 0,
            "eta": learning_rate,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.1,
            "seed": 42
        }

        if is_classification and predicted_feature == "target_class":
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = 3
            xgb_params["eval_metric"] = "mlogloss"
        else:
            xgb_params["objective"] = "reg:squarederror"
            xgb_params["eval_metric"] = loss_function

        # ========= ENTRENAMIENTO =========
        print("🚀 Entrenando modelo...")
        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=epochs,
            evals=[(dtrain, "train")],
            evals_result={},
            verbose_eval=False
        )



        # ========= PREDECIR PARA MAÑANA =========
        last_block = scaled_data[-n_lookback:].reshape(1, -1)
        dpredict = xgb.DMatrix(last_block)
        prediction = model.predict(dpredict)

        if is_classification and predicted_feature == "target_class":
            predicted_class_index = int(np.argmax(prediction[0]))  # 0, 1, 2
            forecast = [-1, 0, 1][predicted_class_index]  # Mapeo inverso
        else:
            forecast_scaled = prediction[0]
            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, predicted_col_idx] = forecast_scaled
            forecast = scaler.inverse_transform(dummy)[0, predicted_col_idx]

        # ========= RESULTADO =========
        print("📈 Forecast para mañana:", forecast)
        model.save_model(modelo_path)
        
    elif arquitectura == "LIGHTGBM":
        n_forecast = 1
        batch_size = 0
        layer_size = 0
        dropout = 0
        porcentaje_validacion = 0
        optimizer = 'NA'

        last_loss = 0
        last_mae = 0
        last_mape = 0
        last_mse = 0
        last_msle = 0

        print(df)
        # --------------------------
        # Escalado
        # --------------------------
        if predicted_feature not in df.columns:
            raise ValueError(f"La columna '{predicted_feature}' no está en el DataFrame")
            exit(0)
        # Obtener el índice de la columna a predecir
        predicted_col_idx = df.columns.get_loc(predicted_feature)

        print("⚙️ Entrenamiento inicial")

        # Validar que la columna existe
        if predicted_feature not in df.columns:
            raise ValueError(f"La columna '{predicted_feature}' no está en el DataFrame")
            exit(0)

        if is_classification:
            print("🔍 Clasificación: no se aplicará escalado.")
            scaler = None
            scaled_data = df.values  # Usar datos tal cual para clasificación
        else:
            # Escalar
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(df)
            



            # Guardar scaler
            joblib.dump(scaler, scaler_path)


        # Preparar datos
        X, Y = [], []
        for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
            X.append(scaled_data[i - n_lookback:i])  # Todas las features
            Y.append(scaled_data[i:i + n_forecast, predicted_col_idx])  # Solo la que se quiere predecir

        X = np.array(X)
        Y = np.array(Y)

        print(Y)
        



        import lightgbm as lgb
        from lightgbm import LGBMClassifier, LGBMRegressor, record_evaluation, log_evaluation

        print(X)
        # === Reconvertir datos a 2D ===
        X_2d = X.reshape(X.shape[0], -1)



        # === Diccionario de parámetros adicionales ===
        extra_params = {
            'learning_rate': learning_rate,
            'max_depth': 6,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'random_state': 42
        }

        # === Inicializar modelo con tipo según clasificación o regresión ===
        if is_classification:
            model = LGBMClassifier(n_estimators=epochs, **extra_params)
            eval_metric = loss_function  # ej: "logloss"
        else:
            model = LGBMRegressor(n_estimators=epochs, **extra_params)
            eval_metric = loss_function

        # === Callback para registrar métricas ===
        evals_result = {}
        callbacks = [
            record_evaluation(evals_result),
            log_evaluation(period=10)  # muestra cada 10 iteraciones
        ]

        # === Entrenamiento ===
        model.fit(
            X_2d, Y,
            eval_set=[(X_2d, Y), (X_2d, Y)],
            eval_metric=eval_metric,
            callbacks=callbacks
        )

        # === Predicción completa (todos los X usados en entrenamiento, para análisis o gráfico) ===
        pred_scaled = model.predict(X_2d)
        print("Predicciones históricas (para análisis):", pred_scaled)

        # === Predicción futura: solo 1 valor usando el último bloque ===
        last_block = scaled_data[-n_lookback:]  # Último bloque de entrada
        last_block = last_block.reshape(1, -1)  # Reconvertir a 2D

        forecast_scaled = model.predict(last_block)[0]  # Valor predicho (escalar)

        # === Desescalar si es regresión ===
        if not is_classification and scaler is not None:
            # Crear dummy con todas las columnas
            dummy = np.zeros((1, scaled_data.shape[1]))
            dummy[0, predicted_col_idx] = forecast_scaled
            forecast = scaler.inverse_transform(dummy)[0, predicted_col_idx]
        else:
            forecast = forecast_scaled

        print("✅ Predicción futura (forecast):", forecast)

        
    else:

        # --------------------------
        # Escalado
        # --------------------------
        if predicted_feature not in df.columns:
            raise ValueError(f"La columna '{predicted_feature}' no está en el DataFrame")
            exit(0)
        # Obtener el índice de la columna a predecir
        predicted_col_idx = df.columns.get_loc(predicted_feature)
        if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
            print("⚙️ Entrenamiento inicial")

            # Validar que la columna existe
            if predicted_feature not in df.columns:
                raise ValueError(f"La columna '{predicted_feature}' no está en el DataFrame")
                exit(0)

            if is_classification:
                print("🔍 Clasificación: no se aplicará escalado.")
                scaler = None
                scaled_data = df.values  # Usar datos tal cual para clasificación
            else:
                # Escalar
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_data = scaler.fit_transform(df)

                # Guardar scaler
                joblib.dump(scaler, scaler_path)

            # Preparar datos
            X, Y = [], []
            for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
                X.append(scaled_data[i - n_lookback:i])  # Todas las features
                Y.append(scaled_data[i:i + n_forecast, predicted_col_idx])  # Solo la que se quiere predecir

            X = np.array(X)
            Y = np.array(Y)

            
                

        
            if arquitectura == "GRU":

                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import GRU, Dense, Dropout, Input
                from tensorflow.keras.optimizers import Adam, SGD, RMSprop

                n_timesteps = X.shape[1]
                n_features = X.shape[2]

                half = max(1, int(layer_size / 2))

                model = Sequential([
                    Input(shape=(n_timesteps, n_features)),
                    GRU(int(layer_size), return_sequences=True),
                    Dropout(dropout),
                    GRU(half, return_sequences=False),
                    Dropout(dropout),
                    Dense(1, activation='sigmoid' if is_classification else 'linear')
                ])

                if optimizer == "adam":
                    opt = Adam(learning_rate=learning_rate)
                elif optimizer == "sgd":
                    opt = SGD(learning_rate=learning_rate, momentum=0.9)
                elif optimizer == "rmsprop":
                    opt = RMSprop(learning_rate=learning_rate)
                else:
                    raise ValueError(f"Optimizador no soportado: {optimizer}")
                    exit(0)

                model.compile(optimizer=opt, loss=loss_function, metrics=['mae', 'mape', 'mse', 'msle'])

                from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

                checkpoint = ModelCheckpoint(
                    filepath='GRU-weights-best.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    verbose=1
                )

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                print(model.summary())

                training = model.fit(
                    X, Y,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0,
                    shuffle=False,
                    callbacks=[checkpoint, early_stop]
                )


                model.save(modelo_path)

    


            elif (arquitectura == "LSTM"):

                # === MODELO LSTM ===
                n_timesteps  = X.shape[1]
                n_features   = X.shape[2]      # <- múltiple

                half = max(1, int(layer_size/2))     # evita 0 unidades si layer_size<2

                model = Sequential([
                    Input(shape=(n_timesteps, n_features)),
                    LSTM(int(layer_size), return_sequences=True),
                    Dropout(dropout),
                    LSTM(half, return_sequences=False),
                    Dropout(dropout),
                    Dense(1, activation='sigmoid' if is_classification else 'linear')
                ])

                from tensorflow.keras.optimizers import Adam, SGD, RMSprop

                if optimizer == "adam":
                    opt = Adam(learning_rate=learning_rate)  # tú eliges el valor
                elif optimizer == "sgd":
                    opt = SGD(learning_rate=learning_rate, momentum=0.9)
                elif optimizer == "rmsprop":
                    opt = RMSprop(learning_rate=learning_rate)
                else:
                    raise ValueError(f"Optimizador no soportado: {optimizer}")
                    exit(0)


                if is_classification:
                    model.add(Dense(1, activation="tanh"))
                    model.compile(optimizer=opt, loss=loss_function, metrics=['mae', 'mape', 'mse', 'msle'])
                else:
                    model.add(Dense(1))
                    model.compile(optimizer=opt, loss=loss_function, metrics=['mae', 'mape', 'mse', 'msle'])



                from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
                #Usar el formato HDF5 (.h5) en lugar del nuevo .keras:
                checkpoint = ModelCheckpoint(
                    filepath='LSTM-weights-best.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,  # Solo guarda los pesos, no la arquitectura
                    mode='min',   # asegura que se guarda el modelo con menor pérdida
                    verbose=1
                )

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                print(model.summary())

                



                #Entrenamos con el modelo con los datos de entrenamiento y dejamos validation_split% para validación 
                #La validación nos dice si hay overfitting del modelo, si con estos hiperparámetros (epochs, batch_size, layer_size...) no hay overfitting, podemos utilizar validation_split = 0 y todo para entrenar
                training = model.fit(
                    X, Y,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split= 0,
                    shuffle=False,  # clave en series temporales
                    callbacks=[checkpoint, early_stop]  
                )


                model.save(modelo_path)

                


                            
            else:
                print("Arquitectura NO válida")
                exit(0)
        
            
        else:
            # --------------------------
            # Entrenamiento incremental
            # --------------------------
            print("➕ Entrenamiento incremental con nueva fecha")

            # Cargar modelo y scaler entrenado con varias columnas
            model = load_model(modelo_path)
            scaler = joblib.load(scaler_path)

            # Escalar los nuevos datos (deben tener las mismas columnas y orden que en el entrenamiento original)
            scaled_data = scaler.transform(df)

            # Obtener el índice de la columna que quieres predecir
            col_idx = df.columns.get_loc(predicted_feature)

            # Calcular la posición para crear X_new y Y_new
            i = len(scaled_data) - n_lookback - n_forecast
            if i >= 0:
                # Preparar secuencia de entrada
                X_new = np.array([scaled_data[i:i + n_lookback]])

                # Salida deseada (solo columna objetivo)
                Y_new = np.array([scaled_data[i + n_lookback:i + n_lookback + n_forecast, col_idx]])

                # Entrenar incrementalmente
                training = model.fit(X_new, Y_new, epochs=epochs, batch_size=batch_size, verbose=1)

                # Guardar modelo actualizado
                model.save(modelo_path)

        
        if (loading_model == "NO"):
            # Delete each file if it exists
            for path_loading in [modelo_path, scaler_path]:
                if os.path.exists(path_loading):
                    os.remove(path_loading)
                    print(f"Deleted: {path_loading}")
                else:
                    print(f"File not found: {path_loading}")

        # Guardamos las métricas del último epoch en variables individuales
        last_loss = training.history['loss'][-1]      # Error cuadrático medio (MSE)
        last_mae = training.history['mae'][-1]        # Error absoluto medio (MAE)
        last_mape = training.history['mape'][-1]      # Error porcentual absoluto medio (MAPE)
        last_mse = training.history['mse'][-1]        # Redundante con 'loss', pero también se guarda
        last_msle = training.history['msle'][-1]      # Error logarítmico cuadrático medio (MSLE)

        # Mostramos los resultados con contexto
        print(f"Último loss (MSE): {last_loss:.4f}  # Error cuadrático medio")
        print(f"Último MAE: {last_mae:.4f}          # Error absoluto medio")
        print(f"Último MAPE: {last_mape:.2f}%       # Error porcentual absoluto medio")
        print(f"Último MSE: {last_mse:.4f}          # Igual que 'loss'")
        print(f"Último MSLE: {last_msle:.4f}        # Error logarítmico cuadrático medio")
        

        

        # ========================
        # PREDICCIÓN FINAL
        # ========================
        #Tomamos los últimos n_lookback pasos (por ejemplo, los últimos 60 días)
        last_block = scaled_data[-n_lookback:]
        #Los redimensionamos a la forma que espera el modelo
        last_block = last_block.reshape(1, n_lookback, scaled_data.shape[1])

        #Le pasas al modelo el bloque (last_block), y devuelve una predicción de n_forecast pasos (ej., los próximos 10 días), pero todavía en escala 0–1
        forecast_scaled = model.predict(last_block)[0].reshape(-1, 1)

        #VAMOS A DESESCALAR
        #Creamos una matriz vacía con la forma (n_forecast, n_features) — todas las columnas necesarias para inverse_transform
        temp = np.zeros((n_forecast, scaled_data.shape[1]))
        #Colocamos las predicciones escaladas solo en la columna que corresponde a la variable predicha (por ejemplo, "Close") y dejamos el resto como ceros porque MinMaxScaler.inverse_transform() necesita todos los features, aunque solo uno tenga valor
        temp[:, predicted_col_idx] = forecast_scaled.flatten()

        #Convertimos los valores predichos desde escala 0–1 a su escala original (por ejemplo, precios de acciones reales). Solo cogemos la columna de interés (predicted_col_idx)
        if is_classification:
            # Clasificación: convertir valores predichos en etiquetas -1 o 1
            forecast = np.where(forecast_scaled >= 0, 1, -1)
        else:
            forecast = scaler.inverse_transform(temp)[:, predicted_col_idx]

        print("forecast =", forecast.tolist())

    print(forecast)
    print("AQUI FORECAST")



    # organize the results in a data frame
    df_past = df[[predicted_feature]].reset_index()
    df_past.rename(columns={'index': 'Date', predicted_feature: 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future = pd.DataFrame() # Initialize df_future if it doesn't exist
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = forecast
    df_future['Actual'] = np.nan

    #results = df_past.append(df_future).set_index('Date')
    results = pd.concat([df_past, df_future]).set_index('Date')
    results['Test'] = np.nan

    print(results)
    print("----")

    # plot the results
    #results.plot(title=stock)

    #---------------------------------------------------
    ultimo_actual = results.iloc[-(n_forecast + 1)].Actual
    primer_forecast = results.iloc[-n_forecast].Forecast
    #OFFSET PORQUE EL VALOR PREDECIDO NO SE AJUSTA AL FINAL
    diferencia = abs(ultimo_actual - primer_forecast)


    if con_diferencia == "NO":
        diferencia = 0

    if (diferencia > 0):
        # Mover la columna 'Forecast' una fila hacia arriba
        results['Forecast'] = results['Forecast'].shift(-1)

    if primer_forecast < ultimo_actual:
        results.Forecast = results['Forecast'] + diferencia
        #results.Forecast[-n_forecast + 1] = 23
    else:
        results.Forecast = results['Forecast'] - diferencia

    results.Forecast[-n_forecast - 1] = results.Actual[-n_forecast - 1]
    #results['Forecast'] = results['Forecast'].shift(-1)
    results.Forecast[-n_forecast - 2] = results.Forecast[-n_forecast - 10]
    #---------------------------------------------------



    # Convertir end_date a datetime
    end_date_dt = pd.to_datetime(end_date)

    # Encontrar la posición de end_date en el índice
    if end_date_dt in df_original.index:
        idx_pos = df_original.index.get_loc(end_date_dt)

        # Asegurarse de que hay suficientes filas
        if idx_pos + n_forecast < len(df_original):
            end_date_test = df_original.index[idx_pos + n_forecast].strftime('%Y-%m-%d')
            print("end_date_test:", end_date_test)
        else:
            print("No hay suficientes filas después de end_date para alcanzar n_forecast =", n_forecast)
            # Delete each file if it exists
            for path in [modelo_path, scaler_path, fecha_a_predecir_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted: {path}")
                else:
                    print(f"File not found: {path}")

            exit(0)
    else:
        print("end_date no está en el índice del DataFrame.")




    dataframe= results


    # === Descargar precios diarios ===
    from datetime import datetime, timedelta

    

    # Asegúrate de que el índice de df_original es datetime si no lo es:
    df_original.index = pd.to_datetime(df_original.index)

    # Definir end_date como datetime
    end_date = pd.to_datetime(end_date)

    # Filtrar las fechas mayores a end_date
    fechas_posteriores = df_original[df_original.index > end_date]
    print(fechas_posteriores)

    # Obtener la primera fecha posterior (el siguiente 'Date')
    if not fechas_posteriores.empty:
        start_date_test = fechas_posteriores.index[0].strftime('%Y-%m-%d')
        print("start_date_test:", start_date_test)
    else:
        print("No hay una fecha posterior en df_original")
        exit(0)

    
    

    
    #Cojo los datos para test
    df_test = df_original.loc[start_date_test:end_date_test, [predicted_feature,'Close', 'Open', 'High', 'Low']]

    df_test = df_test.loc[:, ~df_test.columns.duplicated()]



    for col in df_test.columns:
        results[col] = pd.NA  # o np.nan, si prefieres
        results.loc[results.index[-n_forecast:], col] = df_test[col].values[:n_forecast]
    # Asignar valores de predicted_feature a la columna 'Test'
    results['Test'].iloc[-n_forecast:] = df_test[predicted_feature].values[:n_forecast]

    # Asignar las fechas del índice de df_test al tramo final de results
    results.index.values[-n_forecast:] = df_test.index[:n_forecast]


    #-----------------------------------------------------------------------------
    results.Test[-n_forecast - 1] = results.Actual[-n_forecast - 1]
    #-----------------------------------------------------------------------------


    print(results.tail(30))



        

    fecha_entrada = results.index[-n_forecast-1]
    
    
    fecha_close = pd.to_datetime(fecha_str)
    # Obtener el valor de Close para esa fecha
    ultimo_cierre = df.loc[fecha_close, 'Close']

 
    if (mostrar_plot == "SI"):
        # ==================== MODO PORCENTAJE DE CAMBIO ====================
        if predicted_feature == "porcentaje_cambio":
            # =================== PORCENTAJE_DE_CAMBIO (dark, barras desde 0, fechas claras, sin solapes) ===================
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mtick
            import pandas as pd
            import numpy as np

            # --------- Estética global (dark) ---------
            plt.rcParams.update({
                "figure.dpi": 120,
                "figure.facecolor": "#0B1220",
                "axes.facecolor": "#0F172A",
                "axes.edgecolor": "#334155",
                "axes.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "semibold",
                "axes.titlepad": 10,
                "axes.labelpad": 6,
                "xtick.color": "#E5E7EB",
                "ytick.color": "#E5E7EB",
                "xtick.major.size": 0,
                "ytick.major.size": 0,
                "text.color": "#E5E7EB",
                "axes.grid": True,
                "grid.color": "#334155",
                "grid.linewidth": 0.8,
                "grid.alpha": 0.7,
                "grid.linestyle": "-",
                "font.size": 11,
                "legend.frameon": False,
            })

            # --------- Paleta ---------
            COLOR_ACTUAL   = "#60A5FA"  # azul
            COLOR_FORECAST = "#FBBF24"  # ámbar
            COLOR_TEST     = "#34D399"  # verde
            COLOR_SHADE    = "#94A3B8"  # gris azulado

            # ====== Datos y recorte ======
            results.index = pd.to_datetime(results.index)
            plot_df = results[-(n_lookback + n_forecast):].copy()

            # ====== Figura ======
            fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

            # ---------- Detectar si % está en decimales (0.02=2%) o enteros (2=2%) ----------
            cols_validas = [c for c in ["Actual","Forecast","Test"] if c in plot_df.columns]
            vals = np.concatenate([plot_df[c].astype(float).to_numpy() for c in cols_validas]) if cols_validas else np.array([0.0])
            vals = vals[np.isfinite(vals)]
            max_abs = float(np.max(np.abs(vals))) if vals.size else 0.0
            datos_en_100 = max_abs > 1.5  # True si 2.0 = 2%

            # ---------- Ventana forecast y ancho de barra ----------
            if n_forecast > 0 and len(plot_df) >= n_forecast:
                forecast_start = plot_df.index[-n_forecast]
            else:
                forecast_start = None

            if len(plot_df.index) >= 2:
                diffs = pd.Series(plot_df.index).diff().dropna()
                step = diffs.median()
                width = (step * 0.8) if isinstance(step, pd.Timedelta) else pd.Timedelta(days=0.8)
            else:
                width = pd.Timedelta(days=0.8)

            # ---------- Máscaras ----------
            idx = plot_df.index
            mask_actual   = (forecast_start is None) or (idx <  forecast_start)
            mask_forecast = (forecast_start is None) or (idx >= forecast_start)

            # ---------- Barras ----------
            if "Actual" in plot_df:
                ax.bar(idx[mask_actual], plot_df.loc[mask_actual, "Actual"], width=width,
                    bottom=0, color=COLOR_ACTUAL, alpha=0.85, label="Actual", zorder=2)

            if "Forecast" in plot_df:
                ax.bar(idx[mask_forecast], plot_df.loc[mask_forecast, "Forecast"], width=width,
                    bottom=0, color=COLOR_FORECAST, alpha=0.85, label="Forecast", zorder=3)

            # ---------- Tramo de pronóstico ----------
            if forecast_start is not None:
                ax.axvline(forecast_start, linestyle="--", linewidth=1.2, color=COLOR_SHADE, alpha=0.85)
                ax.axvspan(forecast_start, plot_df.index[-1], color=COLOR_FORECAST, alpha=0.10)

            # ---------- Línea base en 0 ----------
            ax.axhline(0, color="black", linewidth=3.0, alpha=0.95, zorder=4)

            # ---------- Formato porcentaje ----------
            def formato_pct(y):
                return f"{(y if datos_en_100 else y*100):,.2f}%"

            # ---------- Anotar último Forecast (ámbar) ----------
            if "Forecast" in plot_df and mask_forecast.any():
                xF = idx[mask_forecast][-1]
                yF = float(plot_df.loc[xF, "Forecast"])
                ax.annotate(f"Forecast: {formato_pct(yF)}", xy=(xF, yF),
                            xytext=(10, 0), textcoords="offset points",
                            va="center", fontsize=10, color=COLOR_FORECAST,
                            bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155",
                                    alpha=0.95, lw=0.8))
                ax.scatter([xF], [yF], s=35, zorder=4, color="#0F172A",
                        edgecolor=COLOR_FORECAST, linewidth=1.8)

            # ---------- Barra/etiqueta VERDE en forecast_start con el valor REAL (Test o porcentaje_cambio) ----------
            def pick_real_value_on(date):
                for c in ("Test", "porcentaje_cambio"):
                    if c in plot_df.columns and date in plot_df.index:
                        v = pd.to_numeric(pd.Series([plot_df.loc[date, c]]), errors="coerce").iloc[0]
                        if np.isfinite(v):
                            return float(v)
                return None

            if forecast_start is not None:
                yR = pick_real_value_on(forecast_start)
                if yR is not None:
                    ax.bar([forecast_start], [yR], width=width, bottom=0,
                        color=COLOR_TEST, alpha=0.95, zorder=5, label="Real")
                    ax.annotate(f"Real: {formato_pct(yR)}", xy=(forecast_start, yR),
                                xytext=(-10, 0), textcoords="offset points",
                                va="center", ha="right", fontsize=10, color=COLOR_TEST,
                                bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155",
                                        alpha=0.95, lw=0.8))
                    ax.scatter([forecast_start], [yR], s=35, zorder=6, color="#0F172A",
                            edgecolor=COLOR_TEST, linewidth=1.8)

            # ---------- Eje X ----------
            if len(plot_df) >= 2:
                rango_dias = (plot_df.index[-1] - plot_df.index[0]).days
            else:
                rango_dias = 1

            if rango_dias <= 14:
                major = mdates.DayLocator(interval=1)
                fmt   = mdates.DateFormatter('%d %b')
                minor = mdates.HourLocator(interval=12)
            elif rango_dias <= 90:
                major = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
                fmt   = mdates.DateFormatter('%d %b')
                minor = mdates.DayLocator(interval=2)
            elif rango_dias <= 370:
                major = mdates.MonthLocator(interval=1)
                fmt   = mdates.DateFormatter('%b %Y')
                minor = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
            else:
                major = mdates.MonthLocator(interval=3)
                fmt   = mdates.DateFormatter('%b %Y')
                minor = mdates.MonthLocator(interval=1)

            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_locator(minor)
            ax.tick_params(axis='x', which='major', pad=6)
            fig.autofmt_xdate(rotation=0, ha="center")

            # ---------- Eje Y en porcentaje ----------
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=(100.0 if datos_en_100 else 1.0), decimals=1))

            # ---------- Límites verticales simétricos ----------
            if max_abs == 0:
                max_abs = 1.0 if not datos_en_100 else 5.0
            ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)

            # ---------- Etiquetas y leyenda ----------
            ax.set_title(f"Predicción (porcentaje de cambio) desde {fecha_entrada.strftime('%Y-%m-%d')}", color="white")
            ax.set_xlabel("Fecha", color="white")
            ax.set_ylabel("% Cambio", color="white")
            ax.tick_params(axis='x', colors="white")
            ax.tick_params(axis='y', colors="white")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # deduplicar por etiqueta
                uniq = dict(zip(labels, handles))
                ax.legend(list(uniq.values()), list(uniq.keys()),
                        loc="upper center", bbox_to_anchor=(0.5, -0.08),
                        ncol=min(3, len(uniq)), frameon=False, labelcolor="#E5E7EB")

            # ---------- Bordes visibles abajo/izquierda ----------
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("#334155")
                ax.spines[spine].set_linewidth(0.8)

            fig.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

            # ===================================================================



        # ==================== MODO TARGET CLASS ====================
        elif predicted_feature == "target_class":
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mtick
            import pandas as pd
            import numpy as np

            # --------- Estética global (dark) ---------
            plt.rcParams.update({
                "figure.dpi": 120,
                "figure.facecolor": "#0B1220",
                "axes.facecolor": "#0F172A",
                "axes.edgecolor": "#334155",
                "axes.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "semibold",
                "axes.titlepad": 10,
                "axes.labelpad": 6,
                "xtick.color": "#E5E7EB",
                "ytick.color": "#E5E7EB",
                "xtick.major.size": 0,
                "ytick.major.size": 0,
                "text.color": "#E5E7EB",
                "axes.grid": True,
                "grid.color": "#334155",
                "grid.linewidth": 0.8,
                "grid.alpha": 0.7,
                "grid.linestyle": "-",
                "font.size": 11,
                "legend.frameon": False,
            })

            # --------- Paleta ---------
            COLOR_ACTUAL   = "#60A5FA"  # azul
            COLOR_FORECAST = "#FBBF24"  # ámbar
            COLOR_TEST     = "#34D399"  # verde
            COLOR_SHADE    = "#94A3B8"  # gris azulado

            # ====== Datos y recorte ======
            results.index = pd.to_datetime(results.index)
            plot_df = results[-(n_lookback + n_forecast):].copy()

            # ====== Figura ======
            fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

            # ---------- Forecast start y ancho de barra ----------
            if n_forecast > 0 and len(plot_df) >= n_forecast:
                forecast_start = plot_df.index[-n_forecast]
            else:
                forecast_start = None

            if len(plot_df.index) >= 2:
                diffs = pd.Series(plot_df.index).diff().dropna()
                step = diffs.median()
                width = (step * 0.8) if isinstance(step, pd.Timedelta) else pd.Timedelta(days=0.8)
            else:
                width = pd.Timedelta(days=0.8)

            # ---------- Máscaras ----------
            idx = plot_df.index
            mask_actual   = (forecast_start is None) or (idx <  forecast_start)
            mask_forecast = (forecast_start is None) or (idx >= forecast_start)

            # ---------- Barras ----------
            if "Actual" in plot_df:
                ax.bar(idx[mask_actual], plot_df.loc[mask_actual, "Actual"], width=width,
                    bottom=0, color=COLOR_ACTUAL, alpha=0.85, label="Actual", zorder=2)

            if "Forecast" in plot_df:
                ax.bar(idx[mask_forecast], plot_df.loc[mask_forecast, "Forecast"], width=width,
                    bottom=0, color=COLOR_FORECAST, alpha=0.85, label="Forecast", zorder=3)

            # ---------- Tramo de pronóstico ----------
            if forecast_start is not None:
                ax.axvline(forecast_start, linestyle="--", linewidth=1.2, color=COLOR_SHADE, alpha=0.85)
                ax.axvspan(forecast_start, plot_df.index[-1], color=COLOR_FORECAST, alpha=0.10)

            # ---------- Línea base en 0 ----------
            ax.axhline(0, color="black", linewidth=3.0, alpha=0.95, zorder=4)

            # ---------- Formato valor discreto ----------
            def formato_valor(y):
                return f"{int(y)}"

            # ---------- Anotar último Forecast (ámbar) ----------
            if "Forecast" in plot_df and mask_forecast.any():
                xF = idx[mask_forecast][-1]
                yF = float(plot_df.loc[xF, "Forecast"])
                ax.annotate(f"Forecast: {formato_valor(yF)}", xy=(xF, yF),
                            xytext=(10, 0), textcoords="offset points",
                            va="center", fontsize=10, color=COLOR_FORECAST,
                            bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155",
                                    alpha=0.95, lw=0.8))
                ax.scatter([xF], [yF], s=35, zorder=4, color="#0F172A",
                        edgecolor=COLOR_FORECAST, linewidth=1.8)

            # ---------- Barra/etiqueta VERDE en forecast_start con el valor REAL ----------
            def pick_real_value_on(date):
                for c in ("Test", "target_class"):
                    if c in plot_df.columns and date in plot_df.index:
                        v = pd.to_numeric(pd.Series([plot_df.loc[date, c]]), errors="coerce").iloc[0]
                        if np.isfinite(v):
                            return float(v)
                return None

            if forecast_start is not None:
                yR = pick_real_value_on(forecast_start)
                if yR is not None:
                    ax.bar([forecast_start], [yR], width=width, bottom=0,
                        color=COLOR_TEST, alpha=0.95, zorder=5, label="Real")
                    ax.annotate(f"Real: {formato_valor(yR)}", xy=(forecast_start, yR),
                                xytext=(-10, 0), textcoords="offset points",
                                va="center", ha="right", fontsize=10, color=COLOR_TEST,
                                bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155",
                                        alpha=0.95, lw=0.8))
                    ax.scatter([forecast_start], [yR], s=35, zorder=6, color="#0F172A",
                            edgecolor=COLOR_TEST, linewidth=1.8)

            # ---------- Eje X ----------
            if len(plot_df) >= 2:
                rango_dias = (plot_df.index[-1] - plot_df.index[0]).days
            else:
                rango_dias = 1

            if rango_dias <= 14:
                major = mdates.DayLocator(interval=1)
                fmt   = mdates.DateFormatter('%d %b')
                minor = mdates.HourLocator(interval=12)
            elif rango_dias <= 90:
                major = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
                fmt   = mdates.DateFormatter('%d %b')
                minor = mdates.DayLocator(interval=2)
            elif rango_dias <= 370:
                major = mdates.MonthLocator(interval=1)
                fmt   = mdates.DateFormatter('%b %Y')
                minor = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
            else:
                major = mdates.MonthLocator(interval=3)
                fmt   = mdates.DateFormatter('%b %Y')
                minor = mdates.MonthLocator(interval=1)

            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_locator(minor)
            ax.tick_params(axis='x', which='major', pad=6)
            fig.autofmt_xdate(rotation=0, ha="center")

            # ---------- Eje Y: valores enteros -1, 0, 1 ----------
            ax.set_yticks([-1, 0, 1])  # Solo mostrar -1, 0 y 1
            ax.set_ylim(-1.25, 1.25)   # Margen visual extra

            # ---------- Etiquetas y leyenda ----------
            ax.set_title(f"Predicción (clases -1 / 0 / 1) desde {fecha_entrada.strftime('%Y-%m-%d')}", color="white")
            ax.set_xlabel("Fecha", color="white")
            ax.set_ylabel("Clase", color="white")
            ax.tick_params(axis='x', colors="white")
            ax.tick_params(axis='y', colors="white")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                uniq = dict(zip(labels, handles))
                ax.legend(list(uniq.values()), list(uniq.keys()),
                        loc="upper center", bbox_to_anchor=(0.5, -0.08),
                        ncol=min(3, len(uniq)), frameon=False, labelcolor="#E5E7EB")

            # ---------- Bordes visibles abajo/izquierda ----------
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("#334155")
                ax.spines[spine].set_linewidth(0.8)

            fig.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

        
        

        else:
            # ================= CLOSE (dark) con líneas discontinuas en:
            # - último día con Actual (fin datos reales)
            # - inicio del Forecast (primer Forecast no nulo)
            # - fin del Forecast (último Forecast no nulo)
            # + fechas en X bien definidas (dinámicas)
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as mtick
            import pandas as pd

            # ===== Estética global DARK =====
            plt.rcParams.update({
                "figure.dpi": 120,
                "figure.facecolor": "#0B1220",
                "axes.facecolor": "#0F172A",
                "axes.edgecolor": "#334155",
                "axes.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "semibold",
                "axes.titlepad": 10,
                "axes.labelpad": 6,
                "xtick.color": "#E5E7EB",
                "ytick.color": "#E5E7EB",
                "xtick.major.size": 0,
                "ytick.major.size": 0,
                "text.color": "#E5E7EB",
                "axes.grid": True,
                "grid.color": "#334155",
                "grid.linewidth": 0.8,
                "grid.alpha": 0.7,
                "grid.linestyle": "-",
                "font.size": 11,
                "legend.frameon": False,
            })

            # ===== Paleta =====
            COLOR_ACTUAL   = "#60A5FA"   # azul claro
            COLOR_FORECAST = "#FBBF24"   # ámbar
            COLOR_TEST     = "#34D399"   # verde
            COLOR_SHADE    = "#94A3B8"   # gris azulado

            if "Test" in results.columns:
                results.rename(columns={"Test": "Real"}, inplace=True)
                
            # ===== Datos y recorte =====
            results.index = pd.to_datetime(results.index)
            plot_df = results[-(n_lookback + n_forecast):].copy()

            # ===== Figura =====
            fig = plt.figure(figsize=(14, 6), constrained_layout=True)
            ax  = fig.add_subplot(111)

            # --- Helper: línea con halo ---
            def plot_pretty_line(x, y, color, label=None, lw=2.4):
                for w, a in [(lw+4, 0.12), (lw+2, 0.18)]:
                    ax.plot(x, y, color=color, linewidth=w, alpha=a, solid_capstyle="round")
                ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round", label=label)

            # --- Series ---
            if "Actual" in plot_df:
                plot_pretty_line(plot_df.index, plot_df["Actual"], COLOR_ACTUAL, label="Actual")
            if "Forecast" in plot_df:
                plot_pretty_line(plot_df.index, plot_df["Forecast"], COLOR_FORECAST, label="Forecast")
            if "Real" in plot_df:
                plot_pretty_line(plot_df.index, plot_df["Real"], COLOR_TEST, label="Real")

            # --- Líneas verticales: fin Actual, inicio Forecast, fin Forecast ---
            forecast_mask = plot_df["Forecast"].notna() if "Forecast" in plot_df else pd.Series(False, index=plot_df.index)
            actual_mask   = plot_df["Actual"].notna()   if "Actual"   in plot_df else pd.Series(False, index=plot_df.index)

            last_actual_date    = plot_df.index[actual_mask][-1]   if actual_mask.any()   else None   # p.ej. 2010-01-26
            forecast_start_date = plot_df.index[forecast_mask][0]  if forecast_mask.any() else None   # p.ej. 2010-01-27
            forecast_end_date   = plot_df.index[forecast_mask][-1] if forecast_mask.any() else None

            # Línea en el ÚLTIMO día con Actual
            if last_actual_date is not None:
                ax.axvline(last_actual_date, linestyle="--", linewidth=1.6,
                        color=COLOR_SHADE, alpha=0.95, zorder=6)

            # Línea en el INICIO del forecast
            if forecast_start_date is not None:
                ax.axvline(forecast_start_date, linestyle="--", linewidth=1.6,
                        color=COLOR_SHADE, alpha=0.95, zorder=6)

            # Línea en el FIN del forecast
            if forecast_end_date is not None:
                ax.axvline(forecast_end_date, linestyle="--", linewidth=1.6,
                        color=COLOR_SHADE, alpha=0.95, zorder=6)

            # Sombreado del tramo forecast
            if (forecast_start_date is not None) and (forecast_end_date is not None):
                ax.axvspan(forecast_start_date, forecast_end_date, color=COLOR_FORECAST, alpha=0.08, zorder=0)

            # --- Anotar último valor de cada serie (en el extremo derecho mostrado) ---
            for col, c in [("Actual", COLOR_ACTUAL), ("Forecast", COLOR_FORECAST), ("Real", COLOR_TEST)]:
                if col in plot_df and not plot_df[col].empty:
                    x = plot_df.index[-1]
                    y = float(plot_df[col].iloc[-1])
                    ax.scatter([x], [y], s=30, zorder=7, color="#0F172A", edgecolor=c, linewidth=1.6)
                    ax.annotate(f"{col}: {y:,.2f}", xy=(x, y),
                                xytext=(10, 0), textcoords="offset points",
                                va="center", fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#334155", alpha=0.9, lw=0.8))

            # --- Fechas en X: dinámico y legible ---
            if len(plot_df) >= 2:
                rango_dias = (plot_df.index[-1] - plot_df.index[0]).days
            else:
                rango_dias = 1

            if rango_dias <= 14:
                major = mdates.DayLocator(interval=1);         fmt = mdates.DateFormatter('%d %b')
                minor = mdates.HourLocator(interval=12)
            elif rango_dias <= 90:
                major = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1); fmt = mdates.DateFormatter('%d %b')
                minor = mdates.DayLocator(interval=2)
            elif rango_dias <= 370:
                major = mdates.MonthLocator(interval=1);       fmt = mdates.DateFormatter('%b %Y')
                minor = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
            else:
                major = mdates.MonthLocator(interval=3);       fmt = mdates.DateFormatter('%b %Y')
                minor = mdates.MonthLocator(interval=1)

            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_locator(minor)
            ax.tick_params(axis='x', which='major', pad=6)
            fig.autofmt_xdate(rotation=0, ha="center")

            # --- Eje Y ---
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
            ax.margins(x=0.01, y=0.07)

            # --- Etiquetas, leyenda y bordes ---
            ax.set_title(f"Predicción desde {fecha_entrada.strftime('%Y-%m-%d')}")
            ax.set_xlabel("Fecha", color="white")
            ax.set_ylabel("Precio", color="white")
            ax.tick_params(axis='x', colors="white")
            ax.tick_params(axis='y', colors="white")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                        ncol=min(3, len(handles)), frameon=False, labelcolor="#E5E7EB")

            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("#334155")
                ax.spines[spine].set_linewidth(0.8)

            for col, c in [("Actual", COLOR_ACTUAL), ("Forecast", COLOR_FORECAST), ("Real", COLOR_TEST)]:
                if col in plot_df:
                    s = plot_df[col].dropna()
                    if not s.empty:
                        x = s.index[-1]          # último índice válido de esa serie
                        y = float(s.iloc[-1])    # último valor válido

                        ax.scatter([x], [y], s=30, zorder=7, color="#0F172A",
                                edgecolor=c, linewidth=1.6)
                        ax.annotate(f"{col}: {y:,.2f}", xy=(x, y),
                                    xytext=(10, 0), textcoords="offset points",
                                    va="center", fontsize=10,
                                    bbox=dict(boxstyle="round,pad=0.25",
                                            fc="#1E293B", ec="#334155",
                                            alpha=0.9, lw=0.8))
                        fig.tight_layout()

            plt.show(block=False)
            plt.pause(0.1)
            # plt.close(fig)  # <- si tu bucle se queda colgado tras la 1ª figura, descomenta esta línea



    
    results.drop(columns=["Actual"], inplace=True)
    
    if con_diferencia == "SI":
        results = results[-n_forecast:-1]
    if con_diferencia == "NO":
        results = results[-n_forecast:]

    print(end_date)
    
    print(start_date_test,end_date_test)

    print(results)
    

    results.rename(columns={"Test": "Real"}, inplace=True)

    df_real = results.iloc[:, 1:]
    df_forecast = pd.DataFrame(results['Forecast'])


    precio_entrada = ultimo_cierre

    

    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 1 %
    dias_prediccion_1 = 0
    dias_operacion_1 = 0

    df_resultados_forecast_1 = []
    porcentaje_variacion_1 = 1

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_1/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_1/100)

    print(df_forecast)
    print(df_real)
    print("----")

    
    for i1 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i1]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i1 + 1
                break

            
            elif df_forecast.iloc[i1]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i1 + 1
                break
            
            if (i1 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i1 + 1
                break
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i1]['Forecast'] >= porcentaje_variacion_1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i1 + 1
                break

            
            elif df_forecast.iloc[i1]['Forecast'] <= -porcentaje_variacion_1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i1 + 1
                break
            if (i1 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i1 + 1
                break
            

        else:
            if df_forecast.iloc[i1]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i1 + 1
                break

            
            elif df_forecast.iloc[i1]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i1 + 1
                print("AQUIIII")
                break
            if (i1 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i1 + 1
                break

    for j1 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j1]['Close'] >= precio_salida_alcista:
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (df_real.iloc[j1]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j1]['Close'] <= precio_salida_bajista:
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (df_real.iloc[j1]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j1 == len(df_real)-1):
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (df_real.iloc[j1]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j1]['Close'] <= precio_salida_bajista:
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j1]['Close']) / df_real.iloc[j1]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j1]['Close'] >= precio_salida_alcista:
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j1]['Close']) / df_real.iloc[j1]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j1 == len(df_real)-1):
                dias_operacion_1 = j1 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j1]['Close']) / df_real.iloc[j1]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j1] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_1.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j1],
                    'precio_salida': round(df_real.iloc[j1]['Close'], 2),
                    'dias_prediccion': dias_prediccion_1,
                    'dias_operacion': dias_operacion_1,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_1 = pd.DataFrame(df_resultados_forecast_1)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_1}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_1 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_1}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_1.empty:
        if not os.path.exists(csv_file_1):
            df_resultados_forecast_1.to_csv(csv_file_1, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_1.to_csv(csv_file_1, mode='a', index=False, header=False)
            
    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 2 %
    dias_prediccion_2 = 0
    dias_operacion_2 = 0

    df_resultados_forecast_2 = []
    porcentaje_variacion_2 = 2

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_2/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_2/100)


    
    for i2 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i2]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i2 + 1
                break

            
            elif df_forecast.iloc[i2]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i2 + 1
                break
            
            if (i2 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i2 + 1
                break
        
    
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i2]['Forecast'] >= porcentaje_variacion_2:
                prediccion = "ALCISTA"
                dias_prediccion_2 = i2 + 1
                break

            
            elif df_forecast.iloc[i2]['Forecast'] <= -porcentaje_variacion_2:
                prediccion = "BAJISTA"
                dias_prediccion_2 = i2 + 1
                break
            if (i2 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_2 = i2 + 1
                break
            

        else:

            if df_forecast.iloc[i2]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_2 = i2 + 1
                break

            
            elif df_forecast.iloc[i2]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_2 = i2 + 1
                break
            if (i2 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_2 = i2 + 1
                break

    for j2 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j2]['Close'] >= precio_salida_alcista:
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (df_real.iloc[j2]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j2]['Close'] <= precio_salida_bajista:
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (df_real.iloc[j2]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j2 == len(df_real)-1):
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (df_real.iloc[j2]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j2]['Close'] <= precio_salida_bajista:
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j2]['Close']) / df_real.iloc[j2]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j2]['Close'] >= precio_salida_alcista:
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j2]['Close']) / df_real.iloc[j2]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j2 == len(df_real)-1):
                dias_operacion_2 = j2 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j2]['Close']) / df_real.iloc[j2]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j2] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_2.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j2],
                    'precio_salida': round(df_real.iloc[j2]['Close'], 2),
                    'dias_prediccion': dias_prediccion_2,
                    'dias_operacion': dias_operacion_2,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_2 = pd.DataFrame(df_resultados_forecast_2)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_2}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_2 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_2}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_2.empty:
        if not os.path.exists(csv_file_2):
            df_resultados_forecast_2.to_csv(csv_file_2, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_2.to_csv(csv_file_2, mode='a', index=False, header=False)

    

    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 3 %
    dias_prediccion_3 = 0
    dias_operacion_3 = 0

    df_resultados_forecast_3 = []
    porcentaje_variacion_3 = 3

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_3/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_3/100)


    
    for i3 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i3]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i3 + 1
                break

            
            elif df_forecast.iloc[i3]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i3 + 1
                break
            
            if (i3 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i3 + 1
                break

        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i3]['Forecast'] >= porcentaje_variacion_3:
                prediccion = "ALCISTA"
                dias_prediccion_3 = i3 + 1
                break

            
            elif df_forecast.iloc[i3]['Forecast'] <= -porcentaje_variacion_3:
                prediccion = "BAJISTA"
                dias_prediccion_3 = i3 + 1
                break
            if (i3 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_3 = i3 + 1
                break
            

        else:

            if df_forecast.iloc[i3]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_3 = i3 + 1
                break

            
            elif df_forecast.iloc[i3]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_3 = i3 + 1
                break
            if (i3 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_3 = i3 + 1
                break

    for j3 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j3]['Close'] >= precio_salida_alcista:
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (df_real.iloc[j3]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j3]['Close'] <= precio_salida_bajista:
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (df_real.iloc[j3]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j3 == len(df_real)-1):
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (df_real.iloc[j3]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j3]['Close'] <= precio_salida_bajista:
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j3]['Close']) / df_real.iloc[j3]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j3]['Close'] >= precio_salida_alcista:
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j3]['Close']) / df_real.iloc[j3]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j3 == len(df_real)-1):
                dias_operacion_3 = j3 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j3]['Close']) / df_real.iloc[j3]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j3] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_3.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j3],
                    'precio_salida': round(df_real.iloc[j3]['Close'], 2),
                    'dias_prediccion': dias_prediccion_3,
                    'dias_operacion': dias_operacion_3,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_3 = pd.DataFrame(df_resultados_forecast_3)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_3}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_3 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_3}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_3.empty:
        if not os.path.exists(csv_file_3):
            df_resultados_forecast_3.to_csv(csv_file_3, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_3.to_csv(csv_file_3, mode='a', index=False, header=False)





    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 4 %
    dias_prediccion_4 = 0
    dias_operacion_4 = 0

    df_resultados_forecast_4 = []
    porcentaje_variacion_4 = 4

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_4/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_4/100)


    
    for i4 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i4]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i4 + 1
                break

            
            elif df_forecast.iloc[i4]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i4 + 1
                break
            
            if (i4 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i4 + 1
                break
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i4]['Forecast'] >= porcentaje_variacion_4:
                prediccion = "ALCISTA"
                dias_prediccion_4 = i4 + 1
                break

            
            elif df_forecast.iloc[i4]['Forecast'] <= -porcentaje_variacion_4:
                prediccion = "BAJISTA"
                dias_prediccion_4 = i4 + 1
                break
            if (i4 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_4 = i4 + 1
                break
            

        else:

            if df_forecast.iloc[i4]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_4 = i4 + 1
                break

            
            elif df_forecast.iloc[i4]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_4 = i4 + 1
                break
            if (i4 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_4 = i4 + 1
                break

    for j4 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j4]['Close'] >= precio_salida_alcista:
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (df_real.iloc[j4]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j4]['Close'] <= precio_salida_bajista:
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (df_real.iloc[j4]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j4 == len(df_real)-1):
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (df_real.iloc[j4]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j4]['Close'] <= precio_salida_bajista:
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j4]['Close']) / df_real.iloc[j4]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j4]['Close'] >= precio_salida_alcista:
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j4]['Close']) / df_real.iloc[j4]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j4 == len(df_real)-1):
                dias_operacion_4 = j4 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j4]['Close']) / df_real.iloc[j4]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j4] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_4.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j4],
                    'precio_salida': round(df_real.iloc[j4]['Close'], 2),
                    'dias_prediccion': dias_prediccion_4,
                    'dias_operacion': dias_operacion_4,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_4 = pd.DataFrame(df_resultados_forecast_4)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_4}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_4 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_4}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_4.empty:
        if not os.path.exists(csv_file_4):
            df_resultados_forecast_4.to_csv(csv_file_4, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_4.to_csv(csv_file_4, mode='a', index=False, header=False)

    

    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 5 %
    dias_prediccion_5 = 0
    dias_operacion_5 = 0

    df_resultados_forecast_5 = []
    porcentaje_variacion_5 = 5

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_5/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_5/100)


    
    for i5 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i5]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i5 + 1
                break

            
            elif df_forecast.iloc[i5]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i5 + 1
                break
            
            if (i5 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i5 + 1
                break

        elif predicted_feature == "target_class":
            
            if df_forecast.iloc[i5]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i5 + 1
                break

            
            elif df_forecast.iloc[i5]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i5 + 1
                break
            
            if (i5 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i5 + 1
                break
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i5]['Forecast'] >= porcentaje_variacion_5:
                prediccion = "ALCISTA"
                dias_prediccion_5 = i5 + 1
                break

            
            elif df_forecast.iloc[i5]['Forecast'] <= -porcentaje_variacion_5:
                prediccion = "BAJISTA"
                dias_prediccion_5 = i5 + 1
                break
            if (i5 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_5 = i5 + 1
                break
            

        else:

            if df_forecast.iloc[i5]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_5 = i5 + 1
                break

            
            elif df_forecast.iloc[i5]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_5 = i5 + 1
                break
            if (i5 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_5 = i5 + 1
                break

    for j5 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j5]['Close'] >= precio_salida_alcista:
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (df_real.iloc[j5]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j5]['Close'] <= precio_salida_bajista:
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (df_real.iloc[j5]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j5 == len(df_real)-1):
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (df_real.iloc[j5]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j5]['Close'] <= precio_salida_bajista:
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j5]['Close']) / df_real.iloc[j5]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j5]['Close'] >= precio_salida_alcista:
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j5]['Close']) / df_real.iloc[j5]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j5 == len(df_real)-1):
                dias_operacion_5 = j5 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j5]['Close']) / df_real.iloc[j5]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j5] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_5.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j5],
                    'precio_salida': round(df_real.iloc[j5]['Close'], 2),
                    'dias_prediccion': dias_prediccion_5,
                    'dias_operacion': dias_operacion_5,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_5 = pd.DataFrame(df_resultados_forecast_5)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_5}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_5 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_5}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_5.empty:
        if not os.path.exists(csv_file_5):
            df_resultados_forecast_5.to_csv(csv_file_5, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_5.to_csv(csv_file_5, mode='a', index=False, header=False)

    

    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 6 %
    dias_prediccion_6 = 0
    dias_operacion_6 = 0

    df_resultados_forecast_6 = []
    porcentaje_variacion_6 = 6

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_6/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_6/100)


    
    for i6 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i6]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i6 + 1
                break

            
            elif df_forecast.iloc[i6]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i6 + 1
                break
            
            if (i6 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i6 + 1
                break
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i6]['Forecast'] >= porcentaje_variacion_6:
                prediccion = "ALCISTA"
                dias_prediccion_6 = i6 + 1
                break

            
            elif df_forecast.iloc[i6]['Forecast'] <= -porcentaje_variacion_6:
                prediccion = "BAJISTA"
                dias_prediccion_6 = i6 + 1
                break
            if (i6 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_6 = i6 + 1
                break
            

        else:

            if df_forecast.iloc[i6]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_6 = i6 + 1
                break

            
            elif df_forecast.iloc[i6]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_6 = i6 + 1
                break
            if (i6 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_6 = i6 + 1
                break

    for j6 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j6]['Close'] >= precio_salida_alcista:
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (df_real.iloc[j6]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j6]['Close'] <= precio_salida_bajista:
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (df_real.iloc[j6]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j6 == len(df_real)-1):
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (df_real.iloc[j6]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j6]['Close'] <= precio_salida_bajista:
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j6]['Close']) / df_real.iloc[j6]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j6]['Close'] >= precio_salida_alcista:
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j6]['Close']) / df_real.iloc[j6]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j6 == len(df_real)-1):
                dias_operacion_6 = j6 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j6]['Close']) / df_real.iloc[j6]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j6] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_6.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j6],
                    'precio_salida': round(df_real.iloc[j6]['Close'], 2),
                    'dias_prediccion': dias_prediccion_6,
                    'dias_operacion': dias_operacion_6,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_6 = pd.DataFrame(df_resultados_forecast_6)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_6}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_6 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_6}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_6.empty:
        if not os.path.exists(csv_file_6):
            df_resultados_forecast_6.to_csv(csv_file_6, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_6.to_csv(csv_file_6, mode='a', index=False, header=False)

    


    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 7 %
    dias_prediccion_7 = 0
    dias_operacion_7 = 0

    df_resultados_forecast_7 = []
    porcentaje_variacion_7 = 7

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_7/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_7/100)


    
    for i7 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i7]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i7 + 1
                break

            
            elif df_forecast.iloc[i7]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i7 + 1
                break
            
            if (i7 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i7 + 1
                break

        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i7]['Forecast'] >= porcentaje_variacion_7:
                prediccion = "ALCISTA"
                dias_prediccion_7 = i7 + 1
                break

            
            elif df_forecast.iloc[i7]['Forecast'] <= -porcentaje_variacion_7:
                prediccion = "BAJISTA"
                dias_prediccion_7 = i7 + 1
                break
            if (i7 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_7 = i7 + 1
                break
            

        else:

            if df_forecast.iloc[i7]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_7 = i7 + 1
                break

            
            elif df_forecast.iloc[i7]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_7 = i7 + 1
                break
            if (i7 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_7 = i7 + 1
                break

    for j7 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j7]['Close'] >= precio_salida_alcista:
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (df_real.iloc[j7]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j7]['Close'] <= precio_salida_bajista:
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (df_real.iloc[j7]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j7 == len(df_real)-1):
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (df_real.iloc[j7]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j7]['Close'] <= precio_salida_bajista:
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j7]['Close']) / df_real.iloc[j7]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j7]['Close'] >= precio_salida_alcista:
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j7]['Close']) / df_real.iloc[j7]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j7 == len(df_real)-1):
                dias_operacion_7 = j7 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j7]['Close']) / df_real.iloc[j7]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j7] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_7.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j7],
                    'precio_salida': round(df_real.iloc[j7]['Close'], 2),
                    'dias_prediccion': dias_prediccion_7,
                    'dias_operacion': dias_operacion_7,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_7 = pd.DataFrame(df_resultados_forecast_7)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_7}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_7 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_7}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_7.empty:
        if not os.path.exists(csv_file_7):
            df_resultados_forecast_7.to_csv(csv_file_7, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_7.to_csv(csv_file_7, mode='a', index=False, header=False)

    


    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 8 %
    dias_prediccion_8 = 0
    dias_operacion_8 = 0

    df_resultados_forecast_8 = []
    porcentaje_variacion_8 = 8

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_8/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_8/100)


    
    for i8 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i8]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i8 + 1
                break

            
            elif df_forecast.iloc[i8]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i8 + 1
                break
            
            if (i8 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i8 + 1
                break

        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i8]['Forecast'] >= porcentaje_variacion_8:
                prediccion = "ALCISTA"
                dias_prediccion_8 = i8 + 1
                break

            
            elif df_forecast.iloc[i8]['Forecast'] <= -porcentaje_variacion_8:
                prediccion = "BAJISTA"
                dias_prediccion_8 = i8 + 1
                break
            if (i8 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_8 = i8 + 1
                break
            

        else:

            if df_forecast.iloc[i8]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_8 = i8 + 1
                break

            
            elif df_forecast.iloc[i8]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_8 = i8 + 1
                break
            if (i8 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_8 = i8 + 1
                break

    for j8 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j8]['Close'] >= precio_salida_alcista:
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (df_real.iloc[j8]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j8]['Close'] <= precio_salida_bajista:
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (df_real.iloc[j8]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j8 == len(df_real)-1):
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (df_real.iloc[j8]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j8]['Close'] <= precio_salida_bajista:
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j8]['Close']) / df_real.iloc[j8]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j8]['Close'] >= precio_salida_alcista:
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j8]['Close']) / df_real.iloc[j8]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j8 == len(df_real)-1):
                dias_operacion_8 = j8 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j8]['Close']) / df_real.iloc[j8]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j8] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_8.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j8],
                    'precio_salida': round(df_real.iloc[j8]['Close'], 2),
                    'dias_prediccion': dias_prediccion_8,
                    'dias_operacion': dias_operacion_8,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_8 = pd.DataFrame(df_resultados_forecast_8)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_8}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_8 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_8}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_8.empty:
        if not os.path.exists(csv_file_8):
            df_resultados_forecast_8.to_csv(csv_file_8, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_8.to_csv(csv_file_8, mode='a', index=False, header=False)

    



    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 9 %
    dias_prediccion_9 = 0
    dias_operacion_9 = 0

    df_resultados_forecast_9 = []
    porcentaje_variacion_9 = 9

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_9/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_9/100)


    
    for i9 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i9]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i9 + 1
                break

            
            elif df_forecast.iloc[i9]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i9 + 1
                break
            
            if (i9 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i9 + 1
                break

        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i9]['Forecast'] >= porcentaje_variacion_9:
                prediccion = "ALCISTA"
                dias_prediccion_9 = i9 + 1
                break

            
            elif df_forecast.iloc[i9]['Forecast'] <= -porcentaje_variacion_9:
                prediccion = "BAJISTA"
                dias_prediccion_9 = i9 + 1
                break
            if (i9 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_9 = i9 + 1
                break
            

        else:

            if df_forecast.iloc[i9]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_9 = i9 + 1
                break

            
            elif df_forecast.iloc[i9]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_9 = i9 + 1
                break
            if (i9 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_9 = i9 + 1
                break

    for j9 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j9]['Close'] >= precio_salida_alcista:
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (df_real.iloc[j9]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j9]['Close'] <= precio_salida_bajista:
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (df_real.iloc[j9]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j9 == len(df_real)-1):
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (df_real.iloc[j9]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j9]['Close'] <= precio_salida_bajista:
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j9]['Close']) / df_real.iloc[j9]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j9]['Close'] >= precio_salida_alcista:
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j9]['Close']) / df_real.iloc[j9]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j9 == len(df_real)-1):
                dias_operacion_9 = j9 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j9]['Close']) / df_real.iloc[j9]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j9] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_9.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j9],
                    'precio_salida': round(df_real.iloc[j9]['Close'], 2),
                    'dias_prediccion': dias_prediccion_9,
                    'dias_operacion': dias_operacion_9,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_9 = pd.DataFrame(df_resultados_forecast_9)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_9}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_9 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_9}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_9.empty:
        if not os.path.exists(csv_file_9):
            df_resultados_forecast_9.to_csv(csv_file_9, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_9.to_csv(csv_file_9, mode='a', index=False, header=False)

    

    #-----PARA PREDICCIONES DE SUBIDAS O BAJADAS DEL 10 %
    dias_prediccion_10 = 0
    dias_operacion_10 = 0

    df_resultados_forecast_10 = []
    porcentaje_variacion_10 = 10

    precio_salida_alcista = precio_entrada * (1 + porcentaje_variacion_10/100)
    precio_salida_bajista = precio_entrada * (1 - porcentaje_variacion_10/100)


    
    for i10 in range (0, len(df_forecast)):

        if predicted_feature == "target_class":
            
            if df_forecast.iloc[i10]['Forecast'] == 1:
                prediccion = "ALCISTA"
                dias_prediccion_1 = i10 + 1
                break

            
            elif df_forecast.iloc[i10]['Forecast'] == -1:
                prediccion = "BAJISTA"
                dias_prediccion_1 = i10 + 1
                break
            
            if (i10 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_1 = i10 + 1
                break
        
        elif predicted_feature == "porcentaje_cambio":
            
            if df_forecast.iloc[i10]['Forecast'] >= porcentaje_variacion_10:
                prediccion = "ALCISTA"
                dias_prediccion_10 = i10 + 1
                break

            
            elif df_forecast.iloc[i10]['Forecast'] <= -porcentaje_variacion_10:
                prediccion = "BAJISTA"
                dias_prediccion_10 = i10 + 1
                break
            if (i10 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_10 = i10 + 1
                break
            

        else:

            if df_forecast.iloc[i10]['Forecast'] >= precio_salida_alcista:
                prediccion = "ALCISTA"
                dias_prediccion_10 = i10 + 1
                break

            
            elif df_forecast.iloc[i10]['Forecast'] <= precio_salida_bajista:
                prediccion = "BAJISTA"
                dias_prediccion_10 = i10 + 1
                break
            if (i10 == len(df_forecast)-1):
                prediccion = "LATERAL"
                dias_prediccion_10 = i10 + 1
                break

    for j10 in range (0, len(df_real)):
        if (prediccion == "ALCISTA"):
            if df_real.iloc[j10]['Close'] >= precio_salida_alcista:
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (df_real.iloc[j10]['Close'] - precio_entrada) / precio_entrada * 100

                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)
                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j10]['Close'] <= precio_salida_bajista:
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (df_real.iloc[j10]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j10 == len(df_real)-1):
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (df_real.iloc[j10]['Close'] - precio_entrada) / precio_entrada * 100
                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion > 0):
                    resultado = "EXITOSA"
                else:
                    resultado = "PERDIDA"

                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

        
        if (prediccion == "BAJISTA"):
            if df_real.iloc[j10]['Close'] <= precio_salida_bajista:
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j10]['Close']) / df_real.iloc[j10]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "EXITOSA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            elif df_real.iloc[j10]['Close'] >= precio_salida_alcista:
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j10]['Close']) / df_real.iloc[j10]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)

                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': "PERDIDA",
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break

            if (j10 == len(df_real)-1):
                dias_operacion_10 = j10 + 1

                porcentaje_operacion = (precio_entrada - df_real.iloc[j10]['Close']) / df_real.iloc[j10]['Close'] * 100
                porcentaje_apertura = round(((df_real.Open[j10] - precio_entrada) / precio_entrada ) *100 , 2)

                if (porcentaje_operacion < 0):
                    resultado = "PERDIDA"
                else:
                    resultado = "EXITOSA"

                df_resultados_forecast_10.append({
                    'stock': stock,
                    'capitalizacion': cap,
                    'predicted_feature': predicted_feature,
                    'criterio_entrada_salida': criterio_entrada_salida,
                    'direccion_entrada': prediccion,
                    'resultado': resultado,
                    'fecha_entrada': fecha_entrada,
                    'precio_entrada': round(precio_entrada, 2),
                    'fecha_salida': df_real.index[j10],
                    'precio_salida': round(df_real.iloc[j10]['Close'], 2),
                    'dias_prediccion': dias_prediccion_10,
                    'dias_operacion': dias_operacion_10,
                    'arquitectura': arquitectura,
                    'num_indicadores': num_indicadores,
                    'layer_size': layer_size,
                    'dropout': dropout,
                    'n_lookback': n_lookback,
                    'n_forecast': n_forecast,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'longitud_datos': longitud_datos,
                    'last_loss': last_loss,
                    'last_mae': last_mae,
                    'last_mape': last_mape,
                    'last_mse': last_mse,
                    'last_msle': last_msle,
                    'porcentaje_operacion': round(porcentaje_operacion, 2),
                    'porcentaje_apertura': porcentaje_apertura
                })
                break


    # DataFrame con operaciones
    df_resultados_forecast_10 = pd.DataFrame(df_resultados_forecast_10)


    import os

    # Ruta del directorio
    output_dir = f"Resultados/{porcentaje_variacion_10}"
    os.makedirs(output_dir, exist_ok=True)

    # Nombre completo del archivo
    csv_file_10 = os.path.join(
        output_dir,
        f"resultados_forecast_{porcentaje_variacion_10}_{stock}_{predicted_feature}_{num_indicadores}_{loss_function}_{optimizer}_{learning_rate}_{layer_size}_{dropout}_{n_lookback}_{n_forecast}_{epochs}_{batch_size}_{arquitectura}_{criterio_entrada_salida}.csv"
    )

    # Guardar el DataFrame
    if not df_resultados_forecast_10.empty:
        if not os.path.exists(csv_file_10):
            df_resultados_forecast_10.to_csv(csv_file_10, mode='w', index=False, header=True)
        else:
            df_resultados_forecast_10.to_csv(csv_file_10, mode='a', index=False, header=False)


