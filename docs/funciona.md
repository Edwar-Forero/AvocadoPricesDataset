# Documentación Técnica del Código — Sección: Implementación de Modelos Predictivos
**Proyecto:** Predicción del Precio del Aguacate  
**Enfoque:** Explicación técnica del código, decisiones de diseño y respuestas a posibles preguntas de sustentación  

---

## Tabla de Contenidos
1. [Contexto: ¿Por qué Regresión?](#1-contexto-por-qué-regresión)
2. [Definición de Variables y Split](#2-definición-de-variables-y-split)
3. [One-Hot Encoding](#3-one-hot-encoding)
4. [Escalado con StandardScaler](#4-escalado-con-standardscaler)
5. [Reducción de Multicolinealidad](#5-reducción-de-multicolinealidad)
6. [Modelo 1: Regresión Lineal](#6-modelo-1-regresión-lineal)
7. [Modelo 2: Random Forest Regressor](#7-modelo-2-random-forest-regressor)
8. [Modelo 3: Red Neuronal MLP](#8-modelo-3-red-neuronal-mlp)
9. [Modelo 4: Gradient Boosting Regressor](#9-modelo-4-gradient-boosting-regressor)
10. [Evaluación de Modelos — Métricas](#10-evaluación-de-modelos--métricas)
11. [Comparación y Tabla de Resultados](#11-comparación-y-tabla-de-resultados)
12. [Visualizaciones de Conclusiones](#12-visualizaciones-de-conclusiones)
13. [Preguntas Frecuentes en Sustentación](#13-preguntas-frecuentes-en-sustentación)

---

## 1. Contexto: ¿Por qué Regresión?

```
# El notebook incluye esta justificación:
# "Se seleccionó un enfoque de regresión debido a que la variable objetivo
#  (AveragePrice) es una variable continua."
```

**¿Qué significa?**  
Existen dos grandes categorías de problemas supervisados en Machine Learning:
- **Clasificación:** predecir una categoría discreta (ej. "spam / no spam", "perro / gato").
- **Regresión:** predecir un valor numérico continuo (ej. precio, temperatura, distancia).

`AveragePrice` puede tomar cualquier valor en un rango continuo (e.g., $0.44, $1.37, $3.25). No tiene categorías discretas. Por tanto, los modelos de regresión son los apropiados.

**¿Por qué no clasificación?** Para usar clasificación habría que discretizar el precio (ej. "barato", "medio", "caro"), lo que implica perder información valiosa sobre las diferencias numéricas exactas entre precios.

---

## 2. Definición de Variables y Split

### Código
```python
y = df_clean["averageprice"]
X = df_clean.drop(columns=["averageprice"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### ¿Qué hace cada línea?

**`y = df_clean["averageprice"]`**  
Extrae la variable objetivo (lo que queremos predecir) como una Serie de pandas. Esta es la "respuesta correcta" que los modelos intentarán aprender a predecir.

**`X = df_clean.drop(columns=["averageprice"])`**  
Crea la matriz de features (variables de entrada) eliminando la variable objetivo. Si `averageprice` permaneciera en X, el modelo simplemente aprendería a repetirla — trampa trivial que no generaliza.

**`train_test_split(..., test_size=0.2, random_state=42)`**  
Divide los datos en dos subconjuntos:
- **80% entrenamiento (X_train, y_train):** El modelo aprende los patrones aquí.
- **20% prueba (X_test, y_test):** Se usa únicamente para evaluar el modelo final.

**¿Por qué 80/20?** Es la división más común para datasets de tamaño mediano. Da suficientes datos para que el modelo aprenda (14,599 filas) y una muestra representativa para evaluar (3,650 filas).

**¿Por qué `random_state=42`?** El split es aleatorio por defecto. Sin una semilla fija, cada ejecución produciría una partición diferente y resultados distintos. `random_state=42` garantiza que cualquier persona que ejecute el código obtenga exactamente los mismos conjuntos de entrenamiento y prueba.

**⚠️ Regla crítica: el split SIEMPRE antes del preprocesamiento**  
El split debe hacerse antes de aplicar escalado o cualquier transformación que use estadísticos calculados sobre los datos. Si se aplica StandardScaler a todo el dataset antes del split, la media y desviación estándar se calculan incluyendo el conjunto de prueba, lo cual se denomina **data leakage** (filtración de datos). El modelo "vería" indirectamente el test durante el entrenamiento, produciendo evaluaciones optimistas e irreales.

---

## 3. One-Hot Encoding

### Código
```python
X_train = pd.get_dummies(X_train, columns=["type", "region"], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=["type", "region"], drop_first=True)

# Alinear columnas (IMPORTANTE)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```

### ¿Qué es One-Hot Encoding y por qué es necesario?

Los algoritmos de Machine Learning operan con números, no con texto. Las variables `type` (valores: `conventional`, `organic`) y `region` (54 valores distintos como `Albany`, `Atlanta`, `SanFrancisco`, etc.) son categóricas nominales — no tienen un orden numérico natural.

Si se codificaran como números (conventional=0, organic=1), el modelo asumiría que organic es "el doble" de conventional, lo cual no tiene sentido semántico.

El **One-Hot Encoding** crea una columna binaria (0 o 1) por cada categoría:

```
type original:       type_organic:
conventional    →        0
organic         →        1
conventional    →        0
```

Para `region` con 54 valores, se crean 53 columnas binarias (una por región, excepto la primera que se elimina con `drop_first=True`).

### ¿Por qué `drop_first=True`?

Si `type` tiene 2 valores (conventional, organic) y se crean 2 columnas binarias, la segunda columna es perfectamente predecible a partir de la primera: si `type_conventional=0`, entonces `type_organic=1`. Esto crea **multicolinealidad perfecta** (dependencia lineal exacta entre features), lo que hace que la matriz de features sea singular y la Regresión Lineal no pueda invertirla para calcular los coeficientes. Eliminar la primera columna resuelve esto: la categoría eliminada (llamada "categoría de referencia") queda implícita cuando todas las demás columnas son 0.

### ¿Por qué el `.align(join='left')`?

El conjunto de prueba puede contener observaciones de regiones que no aparecen en el conjunto de entrenamiento (por la aleatoriedad del split), o viceversa. Si el split genera un test que no tiene registros de `region_Portland`, esa columna no será creada por `get_dummies` en `X_test`. Al intentar predecir, el modelo encontraría diferente número de columnas y lanzaría un error.

`X_train.align(X_test, join='left', axis=1, fill_value=0)` fuerza a `X_test` a tener exactamente las mismas columnas que `X_train`. Las columnas faltantes en `X_test` se rellenan con 0 (indicando ausencia de esa región en ese registro).

---

## 4. Escalado con StandardScaler

### Código
```python
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# Fit SOLO con train
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Transform test
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

### ¿Qué hace StandardScaler?

Transforma cada variable numérica para que tenga **media = 0** y **desviación estándar = 1**:

```
x_escalado = (x - media) / desviación_estándar
```

Ejemplo: si `total_bags` tiene media=300,000 y desviación=500,000, un valor de 800,000 se transforma en (800,000 − 300,000) / 500,000 = 1.0.

### ¿Por qué es necesario?

Sin escalado, las variables con valores grandes (ej. `4046` con valores de hasta 22 millones) dominan numéricamente sobre variables con valores pequeños (ej. `month` con valores 1–12). Esto afecta:

- **Regresión Lineal:** Los coeficientes se ven distorsionados por la escala.
- **Red Neuronal (MLP):** Las funciones de activación (ReLU, sigmoid) y el cálculo de gradientes asumen que las entradas están en rangos similares. Entradas con escalas muy diferentes provocan gradientes desbalanceados y entrenamiento inestable.
- **Gradient Boosting y Random Forest:** En teoría, los árboles de decisión son **invariantes al escalado** (solo usan umbrales de corte), por lo que el escalado no los beneficia ni perjudica. Se aplica de todos modos por consistencia.

### ¿Por qué `fit_transform` solo en train y `transform` en test?

- **`fit_transform`:** Calcula la media y desviación del conjunto de entrenamiento, luego aplica la transformación. El `scaler` "memoriza" estos valores.
- **`transform`:** Aplica la transformación usando los valores memorizados del entrenamiento, sin recalcular.

Si se usara `fit_transform` en el conjunto de prueba, se calcularían la media y desviación del test, lo que constituye **data leakage**: el proceso de escalado usaría información que en un escenario real no debería conocerse en el momento de predicción.

### ¿Por qué solo se escalan las columnas numéricas?

Las columnas binarias producto del OHE (con valores 0 y 1) ya están en la misma escala reducida. Escalarlas distorsionaría su interpretación binaria (algunos valores quedarían negativos) sin aportar beneficio.

---

## 5. Reducción de Multicolinealidad

### Código
```python
if "total_volume" in X_train.columns:
    X_train = X_train.drop(columns=["total_volume"])
    X_test  = X_test.drop(columns=["total_volume"])
```

### ¿Por qué se elimina `total_volume`?

Como se evidenció en el heatmap del EDA, `total_volume` es la **suma aritmética exacta** de las variables `4046 + 4225 + 4770`. Tener esta suma junto con sus componentes crea **multicolinealidad perfecta** en los tres tipos PLU más la variable suma.

**Impacto en la Regresión Lineal:** La multicolinealidad hace que la matriz de features sea (casi) singular. Al intentar calcular los coeficientes por OLS (que requiere invertir XᵀX), la inversión es numéricamente inestable o imposible. Los coeficientes resultantes tienen varianza muy alta y son poco confiables.

**Impacto en modelos de árboles:** Los árboles de decisión no son afectados matemáticamente, pero incluir variables redundantes aumenta la dimensionalidad innecesariamente y puede dispersar la importancia de variables entre `total_volume` y sus componentes, dificultando la interpretación.

**¿Por qué el `if` condicional?** Es una buena práctica defensiva. Si el nombre de la columna cambiara por el proceso de estandarización de nombres o si la columna ya fue eliminada en otro paso, el código no falla con un error.

---

## 6. Modelo 1: Regresión Lineal

### Código
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
```

### ¿Cómo funciona internamente?

La Regresión Lineal busca los coeficientes `β` que minimizan la **Suma de Errores Cuadráticos (SSE)**:

```
SSE = Σ(yᵢ − ŷᵢ)² = Σ(yᵢ − (β₀ + β₁x₁ᵢ + ... + βₙxₙᵢ))²
```

La solución analítica (OLS — Ordinary Least Squares) es:

```
β = (XᵀX)⁻¹ Xᵀy
```

No hay proceso iterativo: los coeficientes óptimos se calculan directamente en un paso algebraico.

### ¿Por qué es el modelo baseline?

- **Simplicidad:** Es el modelo más sencillo posible para regresión. Si los datos tienen relaciones lineales, lo capturará perfectamente.
- **Referencia:** Su desempeño establece el mínimo esperado. Cualquier modelo más complejo debe superarlo para justificar su mayor costo computacional e interpretativo.
- **Interpretabilidad:** Cada coeficiente `β_i` indica el cambio esperado en el precio por unidad de cambio en la variable `xᵢ`, manteniendo las demás constantes.

### Limitación principal

Asume que la relación entre features y target es **estrictamente lineal**. No puede capturar:
- Interacciones entre variables (ej. el efecto combinado de región + temporada).
- Relaciones curvilíneas o de umbral.

---

## 7. Modelo 2: Random Forest Regressor

### Código
```python
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

### ¿Cómo funciona internamente?

El Random Forest implementa **bagging (Bootstrap Aggregating)** sobre árboles de decisión:

1. **Para cada uno de los 100 árboles:**
   - Se extrae una muestra aleatoria con reemplazo del conjunto de entrenamiento (bootstrap sample, ~63% de los datos únicos).
   - Se entrena un árbol de decisión completo sobre esa muestra.
   - **En cada nodo de división**, solo se considera un subconjunto aleatorio de `√n_features` variables para encontrar la mejor división (esto se llama "feature randomness").
2. **Predicción:** Para una nueva observación, los 100 árboles producen 100 predicciones individuales. La predicción final del Random Forest es el **promedio** de estas 100 predicciones.

### ¿Por qué el Random Forest es robusto?

**Reducción de varianza sin aumentar sesgo:** Un árbol de decisión individual puede sobreajustarse (alta varianza). Al promediar 100 árboles entrenados en datos ligeramente diferentes, el promedio converge hacia la señal real, reduciendo el ruido de cada árbol individual. Este fenómeno se explica por la Ley de los Grandes Números.

**Descorrelación de árboles:** Si todos los árboles usaran las mismas features en los mismos nodos, estarían altamente correlacionados y el promediado no reduciría la varianza. La selección aleatoria de features en cada nodo asegura que los árboles sean suficientemente diferentes.

### ¿Por qué `n_estimators=100`?

100 árboles es el estándar de facto. Más árboles mejoran marginalmente el desempeño pero aumentan el tiempo de cómputo linealmente. La mejora de pasar de 100 a 500 árboles es típicamente menor al 1% en R².

### ¿Por qué es el mejor modelo del proyecto?

- Las relaciones precio-region y precio-tipo no son lineales.
- Hay múltiples interacciones entre variables (e.g., una región orgánica en temporada alta tiene un precio muy diferente de la misma región en temporada baja).
- Los outliers en las variables de volumen no afectan la divisiones de los árboles de la misma manera que afectan los coeficientes lineales.

---

## 8. Modelo 3: Red Neuronal MLP

### Código
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
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
```

### ¿Cómo funciona internamente?

Una red neuronal MLP es una función matemática compuesta de capas:

```
Entrada (60 features)
    ↓
Capa oculta 1: 100 neuronas con activación ReLU
    ↓
Capa oculta 2: 50 neuronas con activación ReLU
    ↓
Capa de salida: 1 neurona (predicción del precio)
```

Cada neurona calcula: `output = ReLU(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)`

El entrenamiento ajusta los pesos `w` y sesgos `b` mediante **backpropagation**: calcula el gradiente del error respecto a cada peso y los actualiza en dirección contraria al gradiente (descenso de gradiente).

### Explicación de hiperparámetros

| Parámetro | Valor | ¿Por qué? |
|---|---|---|
| `hidden_layer_sizes` | (100, 50) | Dos capas: la primera extrae features de alto nivel, la segunda las combina. Arquitectura simple para evitar sobreajuste. |
| `activation` | `relu` | ReLU(x) = max(0,x). Evita el problema del gradiente desvaneciente de sigmoid/tanh. Estándar moderno. |
| `solver` | `adam` | Adaptive Moment Estimation: ajusta la tasa de aprendizaje individualmente por parámetro. Converge más rápido que SGD clásico. |
| `learning_rate_init` | 0.001 | Tasa de aprendizaje inicial. Valor estándar para Adam. Muy alta → inestabilidad, muy baja → convergencia lenta. |
| `alpha` | 0.001 | Regularización L2: añade la penalización `α × Σw²` a la función de pérdida. Reduce el sobreajuste. |
| `early_stopping` | True | Reserva el 10% del train como validación interna. Detiene el entrenamiento si el error de validación no mejora en 10 épocas consecutivas. |
| `max_iter` | 600 | Número máximo de épocas (pasadas completas por los datos). `early_stopping` puede terminar antes. |

### ¿Por qué el MLP obtiene el peor desempeño a pesar del ajuste?

Las redes neuronales requieren grandes volúmenes de datos para generalizar bien. Con ~18,000 registros y ~60 features (muchas binarias), el espacio de entrada es demasiado disperso y el dataset demasiado pequeño para que la red encuentre representaciones internas robustas. Los métodos de ensamble basados en árboles son empíricamente superiores en problemas de datos tabulares de este tamaño.

---

## 9. Modelo 4: Gradient Boosting Regressor

### Código
```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
```

### ¿Cómo funciona internamente?

El Gradient Boosting es fundamentalmente diferente al Random Forest. Construye árboles **secuencialmente**:

1. **Árbol 0 (inicialización):** Predice la media de `y` para todos los registros.
2. **Árbol 1:** Se entrena para predecir los **residuos** (errores) del árbol 0.
3. **Árbol 2:** Se entrena para predecir los residuos del modelo {árbol0 + árbol1}.
4. **... y así hasta el árbol 200.**

La predicción final es:
```
ŷ = árbol₀ + learning_rate × árbol₁ + learning_rate × árbol₂ + ... + learning_rate × árbol₂₀₀
```

El nombre "Gradient Boosting" viene de que los residuos en cada paso son el gradiente negativo de la función de pérdida (MSE), y cada árbol sigue esa dirección.

### Diferencia clave vs Random Forest

| Característica | Random Forest | Gradient Boosting |
|---|---|---|
| Construcción | Árboles en paralelo (independientes) | Árboles secuenciales (cada uno depende del anterior) |
| Estrategia | Reducir varianza (promediado) | Reducir sesgo (corrección iterativa) |
| Velocidad | Rápido (paralelizable) | Más lento (secuencial) |
| Riesgo principal | Sesgo moderado si los árboles individuales son débiles | Sobreajuste si learning_rate es alto |
| Robustez ante outliers | Alta (promediado diluye outliers) | Moderada (depende de subsample) |

### ¿Por qué estos hiperparámetros?

**`learning_rate=0.05` (bajo):** Cada árbol contribuye solo el 5% de su predicción al modelo final. Esto obliga a construir más árboles pero produce un modelo que generaliza mejor. Existe un trade-off documentado: `learning_rate × n_estimators` debe mantenerse aproximadamente constante para el mismo nivel de complejidad.

**`max_depth=4`:** Los árboles en GB son típicamente poco profundos ("stumps" o árboles débiles). Profundidad 4 permite capturar interacciones de hasta 4 variables sin sobreajustar.

**`subsample=0.8`:** Introduce estocasticidad: cada árbol se entrena en el 80% de los datos seleccionados aleatoriamente. Reduce la correlación entre árboles consecutivos y mejora la generalización, similar al bagging del RF.

---

## 10. Evaluación de Modelos — Métricas

### Código
```python
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)
```

### Explicación Detallada de Cada Métrica

#### MAE — Error Absoluto Medio
```
MAE = (1/n) × Σ|yᵢ − ŷᵢ|
```
- **Interpretación:** Promedio de las diferencias absolutas entre el precio real y el predicho, **en dólares**.
- **MAE = 0.088** para Random Forest significa que, en promedio, el modelo se equivoca en **8.8 centavos** por predicción.
- **Ventaja:** Robusto ante outliers (los trata linealmente).
- **Desventaja:** No penaliza más los errores grandes.

#### RMSE — Raíz del Error Cuadrático Medio
```
RMSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]
```
- **Interpretación:** Similar al MAE pero penaliza errores grandes de forma cuadrática. También está en **dólares**.
- **RMSE = 0.124** para Random Forest significa que el error "cuadrático promedio" es de $0.124.
- **Ventaja:** Penaliza más los errores grandes, útil cuando predecir mal un precio alto es más costoso.
- **Relación con MAE:** RMSE ≥ MAE siempre. Si RMSE >> MAE, hay outliers con errores grandes.

#### R² — Coeficiente de Determinación
```
R² = 1 − (SS_residual / SS_total) = 1 − [Σ(yᵢ − ŷᵢ)²] / [Σ(yᵢ − ȳ)²]
```
- **Interpretación:** Proporción de la variabilidad total del precio que es explicada por el modelo.
- **R² = 0.904** para Random Forest: el modelo explica el **90.4% de la varianza** del precio.
- **R² = 0:** El modelo es tan bueno como simplemente predecir la media siempre.
- **R² = 1:** Predicción perfecta (overfitting en train, imposible en test real).
- **R² < 0:** El modelo es peor que predecir la media (extremadamente malo).
- **Desventaja:** No está en las mismas unidades que y, por lo que no indica el error en dólares.

### ¿Por qué se usan las tres juntas?

Usar solo una métrica puede ser engañoso. Ejemplo:
- Un modelo con R² = 0.95 pero RMSE = 0.50 tiene errores grandes en casos extremos.
- Un modelo con MAE bajo pero R² moderado puede predecir bien el precio promedio pero no la variabilidad.
- Las tres juntas dan una visión completa: MAE para el error típico, RMSE para penalizar errores grandes, R² para la capacidad explicativa global.

### ¿Por qué evaluar sobre el conjunto de prueba y no el de entrenamiento?

Evaluar sobre train mide qué tan bien el modelo **memorizó** los datos, no qué tan bien **generaliza**. Un modelo sobreajustado (overfitted) tendrá R² ≈ 1.0 en train pero desempeño malo en test. Solo el desempeño en **datos nunca vistos** (test) refleja el comportamiento real del modelo en producción.

---

## 11. Comparación y Tabla de Resultados

### Código
```python
results = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Random Forest", "Red Neuronal", "Gradient Boosting"],
    "MAE":  [mae_lr, mae_rf, mae_mlp, mae_gb],
    "RMSE": [rmse_lr, rmse_rf, rmse_mlp, rmse_gb],
    "R2":   [r2_lr, r2_rf, r2_mlp, r2_gb]
})
print(results)
```

### ¿Qué hace este bloque?

Construye un DataFrame de pandas con las métricas de los cuatro modelos para facilitar la comparación visual. Es una tabla resumen que consolida los resultados de los bloques de evaluación anteriores.

**Interpretación de resultados típicos:**

| Modelo | MAE | RMSE | R² | Análisis |
|---|---|---|---|---|
| Regresión Lineal | ~0.195 | ~0.257 | ~0.588 | Baseline. Captura relaciones lineales pero no interacciones. |
| Random Forest | ~0.088 | ~0.124 | ~0.904 | Mejor modelo. Robusto, no-lineal, interpetable. |
| Red Neuronal | ~0.204 | ~0.265 | ~0.562 | Peor desempeño. Dataset insuficiente para redes. |
| Gradient Boosting | ~0.088 | ~0.120 | ~0.915 | Competitivo con RF. Mejor sesgo, similar varianza. |

---

## 12. Visualizaciones de Conclusiones

### 12.1 Gráfico de Barras Comparativo (MAE, RMSE, R²)

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
```

**¿Por qué 3 subplots?** Cada métrica tiene una escala diferente (MAE y RMSE en dólares, R² entre 0 y 1) y una dirección de "mejor" opuesta (MAE/RMSE: menor es mejor; R²: mayor es mejor). Combinarlas en un solo gráfico confundiría la lectura.

**¿Por qué colores distintos por modelo?** Permite identificar visualmente el mismo modelo en los tres paneles sin leer el eje X repetidamente.

### 12.2 Predicciones vs Valores Reales (Scatter Plot)

```python
ax.plot(lims, lims, "k--", linewidth=1.5, label="Predicción perfecta")
```

**¿Qué representa la diagonal?** La línea `y = x` (eje y = eje x) representa la predicción perfecta: predicho = real. Cuanto más cerca estén los puntos de esta línea, menor es el error del modelo.

**¿Qué indica la dispersión?** La dispersión de los puntos alrededor de la diagonal indica el error del modelo. Un modelo perfecto tendría todos los puntos sobre la diagonal.

**`alpha=0.3`:** Transparencia del 70% para los puntos. Permite visualizar las zonas de mayor densidad cuando hay muchos puntos superpuestos.

### 12.3 Distribución de Residuos

```python
residuos = y_test.values - y_pred
ax.axvline(0, color="black", linestyle="--")
ax.axvline(residuos.mean(), color="red", linestyle=":")
```

**¿Por qué analizar los residuos?** Los residuos deben cumplir ciertos supuestos para que el modelo sea válido: (1) media ≈ 0 (sin sesgo), (2) distribución simétrica, (3) varianza constante.

**Línea negra en 0:** Referencia de error cero.
**Línea roja:** Media de los residuos. Si se aleja de 0, el modelo tiene sesgo sistemático (sobreestima o subestima consistentemente).

### 12.4 Importancia de Variables

```python
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
```

**`feature_importances_`:** Atributo calculado automáticamente por sklearn durante el entrenamiento del Random Forest. Mide el promedio de reducción de impureza (varianza en regresión) atribuible a cada feature en todos los árboles.

**¿Por qué se muestra solo el top 15?** Con ~60 features, mostrar todas haría el gráfico ilegible. Las 15 más importantes concentran la mayor parte del poder predictivo.

**`sort_values(ascending=False)`:** Ordena de mayor a menor importancia para que la feature más relevante aparezca en la parte superior del gráfico de barras horizontales.

---

## 13. Preguntas Frecuentes en Sustentación

**¿Por qué no se usó validación cruzada?**  
La validación cruzada (k-fold) es la práctica ideal: divide los datos en k grupos y evalúa k veces, usando cada grupo como test una vez. Proporciona estimaciones más confiables del desempeño. En este proyecto se usó un split simple 80/20 por simplicidad y para facilitar la comparación directa entre modelos. Con k-fold, los resultados serían más robustos pero similares en magnitud dado el tamaño del dataset.

**¿Por qué el Random Forest es mejor que la Regresión Lineal?**  
El precio del aguacate tiene relaciones no lineales: el efecto de la región sobre el precio varía dependiendo del tipo de aguacate (interacción). La Regresión Lineal no puede modelar estas interacciones. El Random Forest, al construir árboles de decisión que realizan divisiones recursivas, captura naturalmente estas no-linealidades e interacciones.

**¿Por qué el MLP es peor si las redes neuronales son más "poderosas"?**  
"Más poderoso" no significa "mejor en todos los contextos". Las redes neuronales requieren grandes volúmenes de datos (típicamente cientos de miles o millones de ejemplos) y arquitecturas cuidadosamente diseñadas para superar a los métodos de ensamble en datos tabulares. Con ~18,000 registros y muchas features binarias dispersas, los métodos de ensamble basados en árboles son históricamente superiores.

**¿Qué es data leakage y cómo se evitó?**  
Data leakage ocurre cuando información del conjunto de prueba se filtra al proceso de entrenamiento, produciendo evaluaciones artificialmente optimistas. Se evitó: (1) realizando el split antes del escalado, (2) usando `fit_transform` solo en train y `transform` en test para el StandardScaler, y (3) calculando el OHE de forma separada en train y test, alineando después.

**¿Por qué se eliminó `total_volume` y no las otras variables de volumen?**  
`total_volume = 4046 + 4225 + 4770` es la suma exacta de las otras tres. Tener la suma y sus componentes simultáneamente crea dependencia lineal perfecta. Se eliminó la suma (más redundante) y se mantuvieron los componentes porque estos sí aportan información individual sobre qué tipo de aguacate se vende más en cada mercado.

**¿Por qué `drop_first=True` en el OHE?**  
Para evitar la trampa de la variable dummy. Si una variable categórica tiene k categorías y se crean k columnas binarias, la k-ésima es perfectamente predecible como el complemento de las demás. Esto crea multicolinealidad perfecta que impide que la Regresión Lineal invierta la matriz de features. `drop_first=True` elimina una categoría por variable, dejando k−1 columnas.

**¿Qué significa R² = 0.904?**  
El modelo de Random Forest explica el 90.4% de la variabilidad total del precio del aguacate en el conjunto de prueba. El 9.6% restante corresponde a variabilidad que el modelo no captura: puede deberse a factores no incluidos en el dataset (condiciones climáticas, noticias virales sobre el aguacate, especulación de mercado) o ruido aleatorio inherente al mercado.

**¿Por qué el Gradient Boosting usa `learning_rate=0.05` y no 1.0?**  
Con `learning_rate=1.0`, cada árbol contribuye 100% al modelo. El primer árbol que falle en generalizar arrastrará todo el modelo al sobreajuste. Con `learning_rate=0.05`, cada árbol contribuye poco y el modelo aprende gradualmente, siendo más resistente al ruido. Existe un trade-off documentado: menor learning_rate requiere más árboles (`n_estimators` mayor), pero produce mejor generalización.

**¿Qué es el `early_stopping` del MLP y por qué se usó?**  
`early_stopping=True` divide internamente el conjunto de entrenamiento en 90% para entrenar y 10% para validar. Si el error de validación no mejora durante 10 épocas consecutivas (parámetro `n_iter_no_change`), el entrenamiento se detiene aunque no se hayan completado las `max_iter=600` épocas. Esto previene el sobreajuste: el modelo podría memorizar el conjunto de entrenamiento si se entrena demasiado, perdiendo capacidad de generalización.
