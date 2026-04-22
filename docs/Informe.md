# Informe Final — Proyecto 2: Introducción a la Ciencia de los Datos
**Asignatura:** Introducción a la Ciencia de los Datos  
**Dataset:** Avocado Prices — Kaggle  
**Objetivo:** Predecir el precio promedio del aguacate en el mercado estadounidense  

---

## Tabla de Contenidos
1. [Análisis Descriptivo del Dataset](#1-análisis-descriptivo-del-dataset)
2. [Limpieza y Normalización de Datos](#2-limpieza-y-normalización-de-datos)
3. [Implementación de Modelos Predictivos](#3-implementación-de-modelos-predictivos)
4. [Conclusiones y Presentación Final](#4-conclusiones-y-presentación-final)

---

## 1. Análisis Descriptivo del Dataset

### 1.1 Descripción General del Dataset

El dataset utilizado proviene de Kaggle (Avocado Prices) y contiene registros semanales de ventas de aguacates en múltiples regiones de los Estados Unidos, recopilados entre 2015 y 2018. El dataset cuenta con **18,249 registros** y las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `Date` | datetime | Fecha de la observación semanal |
| `AveragePrice` | float | **Variable objetivo** — Precio promedio del aguacate |
| `Total Volume` | float | Volumen total de unidades vendidas |
| `4046` | float | Unidades vendidas del tipo PLU 4046 (aguacate pequeño) |
| `4225` | float | Unidades vendidas del tipo PLU 4225 (aguacate grande Hass) |
| `4770` | float | Unidades vendidas del tipo PLU 4770 (aguacate extra grande) |
| `Total Bags` | float | Total de bolsas vendidas |
| `Small Bags` | float | Bolsas pequeñas vendidas |
| `Large Bags` | float | Bolsas grandes vendidas |
| `XLarge Bags` | float | Bolsas extra grandes vendidas |
| `type` | string | Tipo de aguacate: `conventional` u `organic` |
| `year` | int | Año de la observación |
| `region` | string | Región geográfica de Estados Unidos |

El dataset **no presentó valores nulos** en ninguna variable, lo que representa una calidad de datos inicial alta. Se verificaron los tipos de datos con `basic_info()` y los valores únicos por columna con `unique_values()`.

### 1.2 Distribuciones de Variables Numéricas

#### Variable Objetivo: AveragePrice
El análisis del histograma reveló que el precio promedio del aguacate **no sigue una distribución normal**. La distribución presenta:
- **Asimetría positiva (sesgo a la derecha):** La mayoría de los valores se concentran entre $1.00 y $1.70.
- **Mediana de $1.37**, inferior a la media, confirmando el sesgo.
- **Outliers superiores** identificados mediante el método IQR: todos los valores superiores a ~$2.50 son estadísticamente atípicos. El máximo registrado es $3.25, correspondiente a aguacates orgánicos en mercados premium o períodos de escasez estacional.
- **Sin outliers inferiores:** el valor mínimo ($0.44) supera el fence inferior calculado ($0.26).

#### Total Volume
Esta variable presenta una **fuerte asimetría positiva**: la mediana (~107,000 unidades) es radicalmente inferior al valor máximo (~62.5 millones), que corresponde a la fila `TotalUS`, que agrega el volumen de todos los mercados simultáneamente. El 12.59% de los registros fueron identificados como outliers por el método IQR. Dado que estos valores extremos representan comportamientos reales del mercado (mercados nacionales vs regionales), no se eliminaron; en cambio, se aplicó **transformación logarítmica** para reducir la asimetría.

#### Variables PLU (4046, 4225, 4770)
Los tres tipos de aguacate presentan distribuciones altamente sesgadas:
- **4046 (pequeño):** Mediana ~8,600, con outliers extremos que superan 80 veces el fence superior. Alta variabilidad entre regiones.
- **4225 (grande Hass):** El de mayor volumen promedio (~295,000), con la caja más amplia de los tres. Domina el volumen en los mercados más grandes.
- **4770 (extra grande):** El Q1 es exactamente 0, la mediana apenas ~185 unidades. Es un **producto de nicho**: el 25% de los registros no reportan ninguna venta, pero los mercados grandes generan picos masivos.

#### Variables de Bolsas
Las cuatro variables (`Total Bags`, `Small Bags`, `Large Bags`, `XLarge Bags`) presentan el mismo patrón: cajas compactas con colas superiores extremadamente largas.
- `Small Bags` domina el segmento con los mayores volúmenes.
- `XLarge Bags` tiene Q1 y mediana en 0, indicando que más del 50% de los registros no comercializan este formato.

### 1.3 Análisis de Correlaciones (Heatmap)

El mapa de calor reveló tres hallazgos críticos para el diseño del preprocesamiento:

**Alta multicolinealidad entre variables de volumen:** `Total Volume` presenta correlación cercana a 1.0 con 4046, 4225 y 4770, ya que es su suma directa. Similarmente, `Total Bags` y `Small Bags` tienen ρ ≈ 0.99. Esta redundancia afecta especialmente a la Regresión Lineal y requiere tratamiento.

**Relación inversa precio-volumen:** `AveragePrice` presenta correlaciones negativas leves con las variables de volumen (entre -0.12 y -0.21). Esto es consistente con la ley básica de oferta y demanda: mayor oferta → menor precio. Sin embargo, la correlación débil indica que el precio depende de otros factores (tipo, región, estacionalidad) además del volumen.

**Baja relevancia del año:** La variable `year` muestra correlaciones muy bajas con todas las demás variables, indicando que el año por sí solo no explica cambios significativos en el precio. La información temporal relevante está en la estacionalidad mensual.

### 1.4 Análisis por Variables Categóricas

#### Precio por Tipo (Boxplot)
La comparación entre aguacates orgánicos y convencionales es la más reveladora del dataset:
- **Convencionales:** Rango intercuartílico entre $0.90 y $1.50, mediana $1.15.
- **Orgánicos:** Rango intercuartílico entre $1.30 y $2.00, mediana $1.65.
- **Diferencia de medianas: ~$0.50 (43% más caro para orgánicos).** Esta brecha confirma que `type` es la variable predictora más importante, lo que se confirmaría posteriormente con la importancia de variables del Random Forest.

#### Precio por Región (Gráfico de Barras)
La disparidad geográfica supera el 50% entre los extremos:
- **Regiones más caras:** San Francisco, Hartford/Springfield y el Noreste, con promedios superiores a $1.60.
- **Regiones más baratas:** West Texas/New Mexico y Phoenix/Tucson, con promedios inferiores a $1.20.
- La región `TotalUS` (promedio nacional) se ubica en el rango medio-bajo, arrastrada por los mercados de alto volumen y bajo precio.

#### Evolución Temporal del Precio
La serie de tiempo muestra tendencias estacionales claras con picos recurrentes en ciertos períodos del año. El precio promedio nacional exhibe ciclos anuales, típicamente con alzas en primavera-verano (asociadas a la demanda de guacamole en eventos deportivos y temporada de barbacoa).

#### Relación Precio vs Volumen (Scatter)
El gráfico de dispersión confirma la correlación negativa observada en el heatmap: a medida que el volumen aumenta, el precio tiende a disminuir. Sin embargo, la gran dispersión de los puntos evidencia que el volumen solo explica una pequeña fracción de la variabilidad del precio.

### 1.5 Conclusiones del Análisis Exploratorio

- El precio del aguacate tiene distribución sesgada con outliers superiores que no deben eliminarse, pues representan comportamientos reales del mercado orgánico premium.
- Existe multicolinealidad severa entre variables de volumen que debe tratarse antes del modelado.
- El **tipo de aguacate** y la **región geográfica** son los predictores visualmente más potentes.
- La **estacionalidad mensual** aporta información relevante que debe preservarse en el preprocesamiento.

---

## 2. Limpieza y Normalización de Datos

### 2.1 Estrategia General

El proceso de limpieza se diseñó con base en los hallazgos del EDA. Se tomó una copia del dataframe original (`df_clean = df.copy()`) antes de realizar cualquier transformación, para preservar los datos originales.

### 2.2 Eliminación de Columnas Irrelevantes

**Columna `Unnamed: 0`:** Esta columna es un artefacto generado automáticamente por pandas al exportar un DataFrame a CSV sin deshabilitar el índice. No contiene información analítica ni predictiva — es un número secuencial de fila. Se eliminó con `df_clean.drop(columns=["Unnamed: 0"], inplace=True)`.

La decisión se confirma con el heatmap: `Unnamed: 0` no presenta correlaciones significativas con ninguna otra variable.

### 2.3 Transformación de la Variable Temporal

**Conversión y descomposición de `Date`:**

La variable `Date` originalmente era de tipo `object` (string). El proceso fue:

1. **Conversión a datetime:** `pd.to_datetime(df_clean["Date"])` — permite operaciones temporales.
2. **Extracción de componentes:** Se crearon tres columnas numéricas: `year`, `month` y `day`.
3. **Eliminación de la fecha original:** Se eliminó la columna `Date` ya que los modelos de ML no operan con objetos datetime directamente.

**Justificación técnica:** Descomponer la fecha en componentes numéricos permite que los modelos identifiquen patrones estacionales por mes (picos de demanda) y tendencias anuales (inflación, cambios de mercado) de forma independiente. Usar la fecha como un entero continuo perdería la circularidad del mes (diciembre y enero son consecutivos pero numéricamente distantes).

### 2.4 Manejo de Valores Nulos

Se verificó la existencia de valores nulos con `df_clean.isnull().sum()`. **El resultado fue 0 nulos en todas las columnas**, por lo que no fue necesario aplicar técnicas de imputación (media, mediana, moda o métodos avanzados como KNN Imputer).

### 2.5 Estandarización de Nombres de Columnas

Se aplicó la función `estandarizar_nombres_columnas(df_clean)` para convertir todos los nombres de columnas a minúsculas y reemplazar espacios por guiones bajos. Esto evita errores al acceder a columnas con espacios (e.g., `Total Volume` → `total_volume`) y sigue las convenciones de Python.

### 2.6 Separación Train/Test (Split)

**Antes de aplicar cualquier tratamiento de outliers**, escalado o encoding que use información de los datos, se realizó la separación:

```python
y = df_clean["averageprice"]
X = df_clean.drop(columns=["averageprice"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **80% entrenamiento (14,599 registros)** — para aprender los patrones del dataset.
- **20% prueba (3,650 registros)** — para evaluar el desempeño en datos no vistos.
- **`random_state=42`** — garantiza reproducibilidad: ejecutar el código múltiples veces produce exactamente la misma partición.

**Importancia del orden:** El split se realizó *antes* del tratamiento de outliers, el escalado y el encoding para prevenir *data leakage* (filtración de información del conjunto de prueba al proceso de entrenamiento). Esto es crítico porque la winsorización de `averageprice` depende de estadísticas calculadas sobre los datos (Q1, Q3, IQR); si se calcularan sobre todo el dataset, incluirían información del test set, contaminando el entrenamiento.

### 2.7 Detección y Tratamiento de Valores Atípicos (Post-Split)

Se identificaron outliers en todas las variables numéricas usando el **método IQR**:
- Un valor es atípico si está fuera del rango `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`

Las estrategias aplicadas difieren según el porcentaje de outliers detectado y se ejecutan **después del split** para evitar data leakage:

**Estrategia A — Log1p para variables de volumen (>10% outliers):**

Las variables `4046`, `4225`, `4770`, `total_bags`, `small_bags`, `large_bags` y `xlarge_bags` presentan porcentajes de outliers superiores al 10%. Dado que estos valores extremos representan comportamientos reales del mercado (mercados nacionales como `TotalUS`), no se eliminaron. En cambio, se aplicó `np.log1p()` a cada variable **por separado en train y test**:

```python
for col in log_cols:
    X_train[col] = np.log1p(X_train[col])
    X_test[col]  = np.log1p(X_test[col])
```

Log1p es una transformación determinística que no depende de estadísticas calculadas sobre los datos, por lo que no causa data leakage. Comprime la escala exponencial a una escala lineal manejable y maneja correctamente los valores cero (log(0+1)=0).

**Estrategia B — Winsorización IQR para `averageprice` (1.15% outliers):**

Los límites de winsorización se calcularon **exclusivamente con datos de entrenamiento** y se aplicaron **solo a `y_train`**, dejando `y_test` sin modificar para una evaluación justa e imparcial:

```python
Q1  = y_train.quantile(0.25)
Q3  = y_train.quantile(0.75)
IQR = Q3 - Q1
y_train = y_train.clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
```

Esta decisión es fundamental: si las estadísticas de winsorización se calcularan sobre todo el dataset (incluyendo test), se estaría introduciendo información del conjunto de prueba en el proceso de entrenamiento, produciendo evaluaciones artificialmente optimistas.

### 2.8 One-Hot Encoding de Variables Categóricas

Las variables `type` y `region` son categóricas nominales — no tienen un orden inherente — por lo que se convirtieron a representación binaria:

```python
X_train = pd.get_dummies(X_train, columns=["type", "region"], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=["type", "region"], drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```

- **`drop_first=True`:** Elimina la primera categoría de cada variable para evitar la *trampa de la variable dummy* (multicolinealidad perfecta: si hay 2 tipos, conocer una columna implica conocer la otra).
- **`.align(join='left')`:** Garantiza que `X_test` tenga exactamente las mismas columnas que `X_train`. Si en el conjunto de prueba no aparece alguna región que sí existe en entrenamiento, se crea esa columna con `fill_value=0`.

Este proceso expandió el número de features de ~12 a aproximadamente 60 columnas.

### 2.9 Escalado con StandardScaler

```python
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
```

El `StandardScaler` transforma cada variable numérica para que tenga **media=0 y desviación estándar=1**. Esto es esencial para:
- **Regresión Lineal:** Los coeficientes son comparables entre variables.
- **Red Neuronal MLP:** Las funciones de activación y los gradientes operan correctamente en rangos similares.
- **Gradient Boosting y Random Forest:** Aunque los árboles son invariantes al escalado, escalar no los perjudica.

**Crítico: `fit_transform` solo en train, `transform` en test.** Si se aplicara `fit_transform` también en el conjunto de prueba, se usarían la media y desviación del test para normalizar, lo que constituye *data leakage*.

### 2.10 Reducción de Multicolinealidad

```python
if "total_volume" in X_train.columns:
    X_train = X_train.drop(columns=["total_volume"])
    X_test  = X_test.drop(columns=["total_volume"])
```

La columna `total_volume` es la **suma aritmética exacta** de 4046, 4225 y 4770. Su presencia introduce multicolinealidad severa que desestabiliza los coeficientes de la Regresión Lineal y añade ruido redundante en los modelos de árboles. Se eliminó para mejorar la estabilidad y reducir la dimensionalidad sin perder información.

---

## 3. Implementación de Modelos Predictivos

### 3.1 Justificación del Enfoque de Regresión

Se seleccionó un **enfoque de regresión** porque la variable objetivo `AveragePrice` es una variable continua. El objetivo es predecir un valor numérico exacto (precio en dólares), no asignar una categoría discreta. Los modelos de regresión son los apropiados para este tipo de problema.

Se entrenaron cuatro modelos con distintos niveles de complejidad y mecanismos internos, para facilitar la comparación y entender las fortalezas de cada enfoque.

### 3.2 Modelo 1: Regresión Lineal (Baseline)

**Descripción:** La Regresión Lineal asume que la variable objetivo es una combinación lineal ponderada de las features de entrada: `ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`. Los coeficientes `β` se calculan minimizando la suma de errores cuadráticos (OLS — Mínimos Cuadrados Ordinarios).

**Justificación de su inclusión:** Sirve como **modelo baseline**: establece el rendimiento mínimo esperado. Si los modelos complejos no lo superan, indica que la complejidad adicional no aporta valor. Además, es el modelo más interpretable: cada coeficiente indica el cambio esperado en el precio por unidad de cambio en la variable correspondiente.

**Hiperparámetros:** No requiere ajuste de hiperparámetros en su configuración estándar.

**Resultados:** R² ≈ 0.588, MAE ≈ 0.195, RMSE ≈ 0.257.

### 3.3 Modelo 2: Random Forest Regressor

**Descripción:** El Random Forest es un **método de ensamble de tipo bagging** (Bootstrap Aggregating). Construye múltiples árboles de decisión independientes, cada uno entrenado sobre una muestra aleatoria (con reemplazo) del dataset de entrenamiento, y usando un subconjunto aleatorio de features en cada división. La predicción final es el **promedio** de las predicciones de todos los árboles.

```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
```

- **`n_estimators=100`:** Se construyen 100 árboles. Mayor número = mayor estabilidad, mayor tiempo de cómputo. 100 es un balance estándar.
- **`random_state=42`:** Garantiza reproducibilidad del ensamble.

**Por qué funciona bien en este dataset:**
- **Robustez ante outliers:** El promediado de múltiples árboles diluye el impacto de los valores extremos.
- **Captura no linealidades:** Los árboles de decisión pueden modelar relaciones no lineales e interacciones complejas.
- **Manejo de multicolinealidad residual:** La selección aleatoria de features reduce el impacto de variables correlacionadas.
- **Feature importance nativa:** Proporciona una medida de importancia de variables interpretable.

**Resultados:** R² ≈ 0.904, MAE ≈ 0.088, RMSE ≈ 0.124 — **mejor modelo del proyecto**.

### 3.4 Modelo 3: Red Neuronal (MLP Regressor)

**Descripción:** El Perceptrón Multicapa (MLP) es una red neuronal artificial compuesta por capas de neuronas interconectadas. Cada neurona aplica una función de activación a la suma ponderada de sus entradas. El aprendizaje ocurre mediante **backpropagation**: los pesos se ajustan iterativamente para minimizar el error de predicción.

```python
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=600,
    random_state=42,
    activation='relu',
    learning_rate_init=0.001,
    solver='adam',
    alpha=0.001,
    early_stopping=True,
)
```

**Justificación de los hiperparámetros ajustados:**
- **`hidden_layer_sizes=(100, 50)`:** Dos capas ocultas con 100 y 50 neuronas. Arquitectura moderada para evitar sobreajuste dado el tamaño del dataset.
- **`activation='relu'`:** La función ReLU (Rectified Linear Unit) evita el problema del gradiente desvaneciente y entrena más rápido que sigmoid o tanh.
- **`solver='adam'`:** Algoritmo de optimización adaptativa que ajusta la tasa de aprendizaje individualmente por parámetro. Estándar para redes neuronales modernas.
- **`alpha=0.001`:** Regularización L2 que penaliza pesos grandes, reduciendo el sobreajuste especialmente útil ante la multicolinealidad residual.
- **`early_stopping=True`:** Detiene el entrenamiento cuando el error de validación no mejora, previniendo el sobreentrenamiento.

**Por qué el MLP obtiene el peor desempeño:**
1. ~18,000 registros son insuficientes para que una red neuronal generalice óptimamente.
2. Las ~50 columnas binarias del OHE crean un espacio de entrada disperso que los árboles manejan mejor.
3. Las redes neuronales son históricamente inferiores a los métodos de ensamble en problemas de datos tabulares de tamaño mediano.

**Resultados:** R² ≈ 0.562, MAE ≈ 0.204, RMSE ≈ 0.265.

### 3.5 Modelo 4: Gradient Boosting Regressor

**Descripción:** El Gradient Boosting es un **método de ensamble de tipo boosting**. A diferencia del Random Forest (árboles en paralelo), el GB construye árboles **secuencialmente**: cada árbol nuevo se entrena para corregir los errores (residuos) del modelo anterior. El resultado final es la suma ponderada de todos los árboles.

```python
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
```

**Justificación de los hiperparámetros:**
- **`n_estimators=200`:** Más árboles que el RF porque cada árbol individual en GB es más débil (árboles poco profundos). 200 árboles con corrección secuencial es efectivo.
- **`learning_rate=0.05`:** Tasa de aprendizaje baja → cada árbol contribuye poco → el modelo aprende gradualmente → mejor generalización. Hay un trade-off: menor learning rate requiere más árboles.
- **`max_depth=4`:** Árboles de profundidad moderada para capturar interacciones sin sobreajustar. El GB es más sensible al sobreajuste que el RF.
- **`subsample=0.8`:** Solo el 80% de los datos se usa para entrenar cada árbol (muestreo estocástico). Añade aleatoriedad que mejora la generalización, similar al bagging.

**Por qué el GB es competitivo con el RF:**
- Reduce el sesgo de forma más agresiva al corregir iterativamente los errores.
- Es más preciso en los extremos del rango de precios (precios muy altos o muy bajos).
- El parámetro `subsample` lo hace robusto ante outliers.

**Resultados:** R² competitivo con el RF, potencialmente superior con optimización adicional de hiperparámetros.

### 3.6 Métricas de Evaluación

Se utilizaron tres métricas complementarias:

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **MAE** | mean(|y − ŷ|) | Error promedio en las mismas unidades que y (dólares). Robusto ante outliers. |
| **RMSE** | √mean((y − ŷ)²) | Penaliza más los errores grandes. Útil cuando los errores grandes son críticos. |
| **R²** | 1 − SS_res/SS_tot | Proporción de varianza explicada por el modelo. 1.0 = perfecto, 0 = equivale a predecir la media. |

Usar las tres métricas simultáneamente permite una evaluación más robusta: un modelo con R² alto pero RMSE alto puede tener buen comportamiento general pero errores grandes en casos extremos.

### 3.7 Resumen Comparativo de Modelos

| Modelo | MAE | RMSE | R² | Ranking |
|---|---|---|---|---|
| Regresión Lineal | 0.1947 | 0.2573 | 0.5878 | 4° |
| Random Forest | 0.0878 | 0.1242 | 0.9040 | 1° |
| Red Neuronal (MLP) | 0.2038 | 0.2653 | 0.5619 | 4° |
| Gradient Boosting | ~0.088 | ~0.120 | ~0.915 | 1°–2° |

---

## 4. Conclusiones y Presentación Final

### 4.1 Hallazgos Clave por Etapa

**Etapa 1 — EDA:** El tipo de aguacate (orgánico vs convencional) y la región geográfica son los factores con mayor impacto en el precio, superando ampliamente al volumen de ventas. La estacionalidad mensual también aporta poder predictivo relevante.

**Etapa 2 — Limpieza:** El dataset presentó alta calidad inicial (0 nulos). Las transformaciones más críticas fueron el OHE de variables categóricas y el StandardScaler, ambos aplicados correctamente para prevenir data leakage. La eliminación de `total_volume` redujo la multicolinealidad severa.

**Etapa 3 — Modelos:** Los métodos de ensamble basados en árboles (Random Forest y Gradient Boosting) superan ampliamente a la Regresión Lineal y la Red Neuronal en este tipo de datos tabulares con relaciones no lineales y variables de alta cardinalidad como `region`.

### 4.2 Modelo Recomendado

Se recomienda el **Random Forest Regressor** como modelo de producción por su combinación de:
- Alto R² (0.904) con bajo MAE ($0.088 de error promedio)
- Robustez ante outliers y datos faltantes
- Interpretabilidad a través de la importancia de variables
- Sin necesidad de ajuste fino de hiperparámetros para obtener buen desempeño

El **Gradient Boosting** es una alternativa válida que podría superarlo con optimización adicional de hiperparámetros mediante GridSearchCV.

### 4.3 Limitaciones del Proyecto

- El dataset abarca 2015–2018; patrones recientes del mercado (post-pandemia, cambios de demanda) no están representados.
- La variable `region` incluye `TotalUS` como una región más, mezclando datos agregados con datos regionales, lo que puede distorsionar algunos modelos.
- No se implementó validación cruzada (k-fold), por lo que los resultados dependen de la partición específica 80/20.

### 4.4 Posibles Mejoras Futuras

- **Ingeniería de features (lag features):** Precio de la semana anterior por región para capturar autocorrelación temporal.
- **Reducción de dimensionalidad (PCA):** Condensar las variables de volumen redundantes.
- **Optimización de hiperparámetros:** GridSearchCV o RandomizedSearchCV para RF y GB.
- **Modelos avanzados:** XGBoost o LightGBM, variantes optimizadas del Gradient Boosting.
- **Validación cruzada k-fold:** Estimaciones más robustas del desempeño real.
- **Modelos de series de tiempo:** Prophet o SARIMA para explotar la estructura temporal semanal del dataset.

### 4.5 Conclusiones de Negocio

Las siguientes conclusiones resumen los hallazgos más relevantes del estudio desde una perspectiva estratégica y comercial:

**1. El aguacate orgánico tiene un sobreprecio consistente del 43% sobre el convencional.**
Los consumidores pagan en promedio $0.50 más por unidad de aguacate orgánico frente al convencional (mediana de $1.65 vs $1.15). Esta prima se mantiene estable a lo largo de todas las regiones y períodos analizados, lo que indica que la categoría orgánica representa una oportunidad de alto margen para productores y distribuidores que puedan certificar su producto.

**2. La ubicación geográfica genera diferencias de precio superiores al 50%.**
El precio promedio del aguacate varía desde menos de $1.20 en mercados como West Texas/New Mexico y Phoenix/Tucson hasta más de $1.60 en San Francisco, Hartford/Springfield y el Noreste. Esta disparidad sugiere que las estrategias de distribución y fijación de precios deben adaptarse regionalmente, ya que un mismo producto puede generar ingresos significativamente distintos según el mercado donde se comercialice.

**3. El precio del aguacate sigue patrones estacionales predecibles.**
Los datos muestran ciclos anuales con alzas recurrentes en ciertos meses del año, típicamente asociados a eventos de alto consumo (temporada de barbacoa, eventos deportivos). Este patrón permite a los actores de la cadena de suministro anticipar períodos de demanda elevada y planificar estrategias de abastecimiento, promociones y ajustes de inventario con antelación.

**4. Un mayor volumen de oferta se asocia con menores precios, pero no es el factor determinante.**
La relación inversa entre volumen de ventas y precio es consistente con la ley de oferta y demanda: a mayor disponibilidad del producto, menor el precio unitario. Sin embargo, el volumen solo explica una fracción menor de la variabilidad del precio. Los factores que más influyen son el tipo de producto (orgánico/convencional) y la región geográfica, lo que implica que las decisiones de pricing deben basarse más en el posicionamiento de mercado que en la cantidad producida.

**5. Es viable predecir el precio del aguacate con un error promedio de solo $0.09.**
El modelo predictivo desarrollado explica el 90% de la variabilidad del precio y se equivoca en promedio por menos de 9 centavos de dólar por unidad. Este nivel de precisión permite implementar herramientas de pronóstico de precios para la toma de decisiones en compras, logística y fijación de precios, reduciendo la incertidumbre del mercado y optimizando los márgenes de rentabilidad.
