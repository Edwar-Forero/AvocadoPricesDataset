# 👥 Integrantes del Grupo

| Nombre Completo       | Código  | Rol            | Correo Electrónico       |
|-----------------------|---------|----------------|--------------------------|
| Edwar Yamir Forero Blanco        | 2559741  | Colaborador  | edwar.forero@correounivalle.edu.co  |
| Faber Alexis Solis Gamboa        | 2559753  | Colaborador  | faber.solis@correounivalle.edu.co   |
| Jhojan Serna Henao               | 2259504  | Colaborador  | jhojan.serna@correounivalle.edu.co  |
| Kevin Hinojosa Osorio            | 2259470  | Colaborador  | kevin.hinojosa@correounivalle.edu.co|


# Desarrollo del proyecto:

## 1. Diseño Del Modelo de Bodega de Datos:

En esta primera sección se estudio el dataset proporcionado, el cual contiene información sobre transacciones de ventas de 10 centros comerciales. Cuenta con un total de *99457* filas y 8 columnas, las cuales son: `invoice_no`, `customer_id`, `gender`, `age`, `category`, `quantity`, `price`, `payment_method`, `invoice_date` y `shopping_mall`. 

Al momento de analizar el dataset se identificaron las siguientes tablas de dimensiones: 

![alt text](/docs/Images/modeloBD.jpg)

Se decidio utilizar un modelo estrella para la bodega de datos porque de acuerdo al análisis del dataset, se identificaron claramente una tabla de hechos (sales) y varias tablas de dimensiones (customer, category, payment, mall y date), Ademas no tenia jerarquias, subcategorias o relaciones complejas, lo que hace que el modelo estrella sea una opción adecuada para este caso.

### Script SQL para crear las tablas en la base de datos:
```sql

CREATE TABLE public.table_category (
  category_id bigint,
  category text
);
CREATE TABLE public.table_customer (
  customer_id text,
  gender text,
  age bigint,
  age_group text
);
CREATE TABLE public.table_date (
  date_id bigint,
  invoice_date timestamp without time zone,
  day integer,
  month integer,
  year integer
);
CREATE TABLE public.table_mall (
  mall_id bigint,
  shopping_mall text
);
CREATE TABLE public.table_payment (
  payment_id bigint,
  payment_method text
);
CREATE TABLE public.table_sales (
  invoice_no text,
  customer_id text,
  category_id bigint,
  payment_id bigint,
  mall_id bigint,
  date_id bigint,
  quantity bigint,
  price double precision,
  total_amount double precision
);
```

## 2. Extracción, Transformación y Carga de Datos (ETL):
En esta sección como primer paso para se realizo la extracción de los datos del dataset utilizando la biblioteca `pandas` de Python. Se cargaron los datos en un DataFrame para su posterior manipulación y análisis.

El proceso de extracción se realizo utilizando la función `pd.read_csv()`. Una vez cargados los datos, se realizo una inspección inicial del DataFrame identificando nombres de columnas y tipos de datos presentes.

En cuanto a la transformación de los se llevo a a cabo los siguientes pasos:

1. Se empezo con la transformación de los datos, en primer lugar  se hizo la transformación de la columna `invoice_date` a un formato de fecha utilizando `pd.to_datetime()`. Despues se creo una nueva tabla llamada `table_date`, la cual se puso para que no tenga duplicados y asi tener solo una fila por cada fecha, luego se hizo un reseteo de index, el cual se le asigno a la columna `date_id`. Tambien, se crea una columna por dia, mes y año, para facilitar el análisis temporal de los datos.

2. Como segundo paso la transformación se generó una nueva columna llamada `total_amount`, la cual representa el valor total de cada transacción, calculado como el producto de la cantidad (`quantity`) y el precio (`price`).

3. Como tercer paso se creo una nueva tabla llamada `table_customer` que contiene información única de cada cliente, incluyendo su `customer_id`, `gender` y `age`. Ademas se creo una nueva columna llamada `age_group`, esta función asigna a cada cliente a un grupo de edad específico según los rangos definidos. Se decidio realizar asi ya que esto facilita el análisis de los datos por grupos de edad, permitiendo identificar patrones y tendencias en el comportamiento de compra de los clientes según su grupo etario.

4. Como cuarto paso se creo una nueva tabla llamada `table_category`, la cual se puso para que no tenga duplicados y asi tener solo una fila por cada categoria, luego se hizo un reseteo de index, el cual se le asigno a la columna `category_id`.

5. Como quinto paso se creo una nueva tabla llamada `table_payment`, la cual se puso para que no tenga duplicados y asi tener solo una fila por cada tipo de pago, luego se hizo un reseteo de index, el cual se le asigno a la columna `payment_id`.

6. Como sexto paso se creo una nueva tabla llamada `table_mall`, la cual se puso para que no tenga duplicados y asi tener solo una fila por cada centro comercial, luego se hizo un reseteo de index, el cual se le asigno a la columna `mall_id`.

7. Como septimo paso se hizo un merge de las tablas `table_category`, `table_mall` y `table_payment` con el DataFrame original `df` utilizando la función `merge()`. Esto se hizo para agregar la información de categoría, centro comercial y método de pago a cada transacción en el DataFrame original. Luego se creó una nueva columna llamada `date_id`, que se calculó a partir de la columna `invoice_date` utilizando el formato 'YYYYMMDD' y se convirtió a tipo entero.

8. Como ultimo paso se hizo un merge de las tablas `table_category`, `table_mall` y `table_payment` con el DataFrame original `df` utilizando la función `merge()`. Esto se hizo para agregar la información de categoría, centro comercial y método de pago a cada transacción en el DataFrame original. Luego se creó una nueva columna llamada `date_id`, que se calculó a partir de la columna `invoice_date` utilizando el formato 'YYYYMMDD' y se convirtió a tipo entero.

En cuanto a  la carga de datos se decidio hacerlo en Supabase utilizando la biblioteca `sqlalchemy` para establecer la conexión con la base de datos y cargar los datos transformados en las tablas correspondientes. Se utilizo la función `to_sql()` de pandas para cargar cada tabla en la base de datos, especificando el nombre de la tabla, el motor de conexión y el comportamiento en caso de que la tabla ya exista (en este caso, se reemplaza).

![alt text](/docs/Images/image.png)

## 3. Consultas Analíticas en SQL:

En esta sección se diseñaron consultas SQL sobre la bodega de datos construida en PostgreSQL, con el objetivo de responder preguntas de negocio relevantes para el dataset de compras en centros comerciales.

### Consulta 1 – Total de ventas por categoría de producto
**Pregunta de negocio:** ¿Qué categorías de productos generan más ingresos?

```sql
SELECT 
    c.category,
    SUM(s.total_amount) AS total_ventas,
    SUM(s.quantity) AS total_unidades
FROM table_sales s
JOIN table_category c ON s.category_id = c.category_id
GROUP BY c.category
ORDER BY total_ventas DESC;
```
**Resultado**
![alt text](/docs/Images/consulta1.png)


**Explicación:** Se une la tabla de hechos con la dimensión de categoría, luego se agrupa por categoría para calcular el total de ingresos `total_amount` y el total de unidades vendidas `quantity`. El resultado se ordena de mayor a menor venta.


### Consulta 2 – Clientes con mayor volumen de compras

**Pregunta de negocio:** ¿Quiénes son los 10 clientes que más han gastado?

```sql
SELECT 
    s.customer_id,
    cu.gender,
    cu.age_group,
    SUM(s.total_amount) AS total_compras,
    COUNT(s.invoice_no) AS num_transacciones
FROM table_sales s
JOIN table_customer cu ON s.customer_id = cu.customer_id
GROUP BY s.customer_id, cu.gender, cu.age_group
ORDER BY total_compras DESC
LIMIT 10;
```
**Resultado**

![alt text](/docs/Images/Consulta2.png)

**Explicación:** Se une la tabla de hechos con la dimensión de clientes, agrupando por cliente para sumar el total gastado y contar el número de transacciones realizadas. Se limita el resultado a los 10 clientes con mayor gasto.


### Consulta 3 – Métodos de pago más utilizados

**Pregunta de negocio:** ¿Cuál es el método de pago preferido por los clientes?

```sql
SELECT 
    p.payment_method,
    COUNT(s.invoice_no) AS veces_usado,
    SUM(s.total_amount) AS total_facturado
FROM table_sales s
JOIN table_payment p ON s.payment_id = p.payment_id
GROUP BY p.payment_method
ORDER BY veces_usado DESC;
```
**Resultado**

![alt text](/docs/Images/Consulta3.png)

**Explicación:** Se une la tabla de hechos con la dimensión de métodos de pago, agrupando por método para contar cuántas veces fue utilizado y el total facturado con cada uno.


### Consulta 4 – Comparación de ventas por mes

**Pregunta de negocio:** ¿En qué meses se concentran más las ventas?

```sql
SELECT 
    d.year,
    d.month,
    SUM(s.total_amount) AS total_ventas,
    COUNT(s.invoice_no) AS num_transacciones
FROM table_sales s
JOIN table_date d ON s.date_id = d.date_id
GROUP BY d.year, d.month
ORDER BY d.year, d.month;
```
**Resultado**

![alt text](/docs/Images/Consulta4-1.png)
![alt text](/docs/Images/Consulta4-2.png)
![alt text](/docs/Images/Consulta4-3.png)

**Explicación:** Se une la tabla de hechos con la dimensión de fechas, agrupando por año y mes para comparar el comportamiento de las ventas a lo largo del tiempo.


### Consulta 5 – Género que compra más

**Pregunta de negocio:** ¿El género masculino o femenino realiza más compras?

```sql
SELECT 
    cu.gender,
    COUNT(s.invoice_no) AS num_transacciones,
    SUM(s.total_amount) AS total_compras
FROM table_sales s
JOIN table_customer cu ON s.customer_id = cu.customer_id
GROUP BY cu.gender
ORDER BY total_compras DESC;
```
**Resultado**

![alt text](/docs/Images/Consulta5.png)


**Explicación:** Se agrupa por género para comparar tanto el número de transacciones como el monto total gastado, permitiendo identificar qué segmento representa mayor volumen de negocio.


### Consulta 6 – Rango de edades que compra más

**Pregunta de negocio:** ¿Qué grupo etario tiene mayor participación en las compras?

```sql
SELECT 
    cu.age_group,
    COUNT(s.invoice_no) AS num_transacciones,
    SUM(s.total_amount) AS total_compras
FROM table_sales s
JOIN table_customer cu ON s.customer_id = cu.customer_id
GROUP BY cu.age_group
ORDER BY total_compras DESC;
```
**Resultado**

![alt text](/docs/Images/Consulta6.png)

**Explicación:** Se agrupa por rango de edad `age_group` para identificar cuál segmento etario realiza más transacciones y genera mayores ingresos al negocio.

## 4. Análisis Descriptivo y Visualización de Datos:

En esta sección se realizó el análisis descriptivo del dataset mediante visualizaciones construidas con las bibliotecas `matplotlib` y `seaborn`. El objetivo fue identificar patrones, tendencias y comportamientos relevantes sobre las ventas, los clientes y los centros comerciales.

---

### 4.1 Ventas por Categoría

Se agruparon las ventas totales (`total_amount`) por categoría de producto y se graficaron en un diagrama de barras.

**Hallazgo:** La categoría *Clothing* (vestimenta) concentra el mayor volumen de ingresos, seguida por *Shoes* (calzado) y *Technology* (tecnología). Esto indica que la vestimenta es la categoría con mayor demanda entre los clientes. El negocio podría beneficiarse fortaleciendo el inventario en esta categoría, asegurando variedad y disponibilidad de productos para maximizar las ventas.

---

### 4.2 Ventas por Método de Pago

Se agruparon las ventas totales por método de pago y se representaron en un diagrama de barras.

**Hallazgo:** El método de pago más utilizado es el *efectivo (Cash)*, seguido por las *tarjetas de crédito (Credit Card)*, mientras que las *tarjetas débito (Debit Card)* presentan la menor participación en las transacciones. Esto refleja que los clientes mantienen una fuerte preferencia por los pagos directos. Sin embargo, la relevancia de las tarjetas de crédito sugiere que existe una oportunidad para diseñar convenios con entidades financieras o programas de beneficios que incentiven su uso.

---

### 4.3 Ventas por Mes

Se extrajo el período mensual a partir de la columna `invoice_date` y se graficó la evolución de las ventas totales a lo largo del tiempo mediante una gráfica de línea.

**Hallazgo:** Se identificó una disminución en las ventas durante los primeros meses del año, especialmente en enero y febrero. Hacia el final del año, particularmente en octubre, noviembre y diciembre, se observa un incremento en el volumen de ventas, asociado a las temporadas festivas. Esta estacionalidad es un indicador clave para planificar inventarios, campañas promocionales y estrategias de abastecimiento anticipado.

---

### 4.4 Distribución de Edad de los Clientes

Se graficó un histograma con estimación de densidad (KDE) sobre la columna `age` para analizar cómo se distribuyen las edades de los clientes.

**Hallazgo:** La distribución muestra una mayor concentración de compradores en el rango de *20 a 40 años*, con una frecuencia especialmente alta entre los 25 y 35 años. Este perfil corresponde a adultos jóvenes y adultos en etapa productiva, lo que permite identificar con mayor precisión el público objetivo del negocio y orientar las estrategias de marketing y fidelización hacia este segmento.

---

### 4.5 Top 10 Clientes con Mayor Volumen de Compras

Se calculó el total acumulado de compras por cliente (`customer_id`), se ordenó de forma descendente y se representaron los 10 clientes con mayor gasto en un diagrama de barras.

**Hallazgo:** Un pequeño grupo de clientes concentra una proporción significativa del total de ventas. Esta distribución asimétrica indica que existen *clientes de alto valor* para el negocio. Identificar y retener a estos clientes mediante estrategias de fidelización, descuentos exclusivos o programas de recompensas puede tener un impacto considerable en los ingresos totales.

---

### 4.6 Tendencia de Ventas por Categoría a lo Largo del Tiempo

Se agruparon las ventas por período mensual y categoría de producto, generando una gráfica de líneas con una serie temporal por cada categoría.

**Hallazgo:** El gráfico permite observar las tendencias estacionales por categoría. Se aprecia que la categoría de vestimenta mantiene consistentemente los valores más altos, con algunos picos en determinadas épocas del año. Otras categorías como tecnología y calzado también muestran variaciones que pueden asociarse a eventos comerciales o temporadas específicas. Esta visualización es especialmente útil para planificar el abastecimiento e inventario por categoría según la época del año.

---

### 4.7 Ventas por Categoría y Método de Pago (Mapa de Calor)

Se construyó una tabla pivote con las ventas totales cruzando categoría de producto y método de pago, y se representó mediante un mapa de calor (*heatmap*) con la biblioteca `seaborn`.

**Hallazgo:** En todas las categorías, el *efectivo* sigue siendo el método de pago dominante. Sin embargo, en la categoría de *Technology* se observa una participación relativa más alta de las tarjetas de crédito en comparación con las demás categorías. Esto sugiere que, para compras de mayor valor unitario como artículos tecnológicos, los clientes prefieren financiarse con crédito en lugar de pagar de contado. Esta información puede servir para diseñar promociones diferenciadas por categoría y tipo de pago.

---

### 4.8 Ventas Totales por Centro Comercial

Se agruparon las ventas totales por centro comercial (`shopping_mall`), ordenándolas de forma descendente, y se graficaron en un diagrama de barras.

**Hallazgo:** Los tres centros comerciales con mayores ingresos son *Mall of Istanbul*, *Kanyon* y *Metrocity*. Esta concentración de ventas en pocos establecimientos sugiere que existen factores diferenciales (ubicación, afluencia de público, mix comercial) que los hacen más productivos. El negocio podría analizar las características de estos centros para replicar sus prácticas exitosas en los establecimientos con menor rendimiento.

---

### 4.9 Ventas por Centro Comercial por Año

Se agruparon las ventas por año y centro comercial, generando un gráfico de barras agrupadas que permite comparar el desempeño de cada establecimiento en los años *2021*, *2022* y *2023*.

**Hallazgo:** A lo largo de los tres años analizados, *Mall of Istanbul* y *Kanyon* lideran de manera consistente las ventas anuales. Se observa cierta estabilidad en la posición relativa de los centros comerciales, sin cambios drásticos de liderazgo entre años. Esto indica una tendencia consolidada en las preferencias de compra de los clientes con respecto a los centros comerciales.

---

## 5.  Conclusiones y presentación final:

El desarrollo de este proyecto permitió aplicar los conceptos fundamentales relacionados con la construcción de una bodega de datos y el análisis de información orientado a la toma de decisiones.

Durante el proyecto se diseñó un modelo de bodega de datos basado en un modelo estrella, el cual permitió organizar la información de manera estructurada mediante una tabla de hechos y varias tablas de dimensiones. Este tipo de modelo facilita la realización de consultas analíticas y mejora la eficiencia en el análisis de grandes volúmenes de datos.

Posteriormente se implementó un proceso ETL (Extracción, Transformación y Carga) utilizando Python, donde se extrajeron los datos del dataset original, se transformaron para adaptarlos al modelo dimensional definido y finalmente se cargaron en una base de datos PostgreSQL.

A través de consultas SQL y visualizaciones de datos se logró analizar el comportamiento de las ventas desde diferentes perspectivas, permitiendo identificar patrones relevantes como las categorías de productos con mayor volumen de ventas, los clientes con mayor actividad de compra y los métodos de pago más utilizados.

Finalmente, el uso de herramientas de análisis y visualización permitió transformar los datos en información útil para la toma de decisiones, demostrando la importancia de las bodegas de datos y del análisis de información dentro de los procesos de inteligencia de negocios.

En cuanto a los hallazgos al analizar los datos, se identificó que existen patrones claros en el comportamiento de consumo de los clientes. En primer lugar, las ventas no se distribuyen de manera uniforme entre todas las categorías, sino que algunas concentran una mayor participación, especialmente la categoría de vestimenta, lo que indica que este tipo de producto tiene una alta demanda y representa una oportunidad importante para enfocar estrategias comerciales, de inventario y promociones.

También se observó que el método de pago más utilizado por los clientes es el efectivo, seguido por la tarjeta de crédito, mientras que la tarjeta débito presenta una menor participación. Esto permite inferir que los consumidores aún mantienen una fuerte preferencia por pagos directos, aunque en compras de mayor valor algunas categorías muestran una mayor presencia de tarjetas de crédito. Este hallazgo puede ser útil para diseñar campañas, beneficios financieros o alianzas con entidades de pago.

En cuanto al comportamiento temporal, las ventas muestran variaciones a lo largo de los meses, lo que sugiere la existencia de estacionalidad. Se evidencian periodos con menor actividad comercial al inicio del año y un aumento en temporadas específicas, especialmente hacia fin de año, lo que puede relacionarse con fechas festivas, vacaciones o eventos comerciales. Este patrón resulta relevante para planificar inventarios, campañas publicitarias y estrategias de abastecimiento.

Respecto al perfil de los clientes, la distribución de edades muestra una mayor concentración de compradores en rangos intermedios, particularmente entre adultos jóvenes y adultos. Esto sugiere que este segmento representa el principal público objetivo del negocio, por lo que las decisiones de mercadeo, segmentación y fidelización pueden orientarse principalmente hacia ese grupo etario.

Adicionalmente, el análisis de clientes permitió identificar que un grupo reducido concentra una parte importante del valor total de compras. Esto evidencia que existen clientes de alto valor para el negocio, y por tanto sería conveniente implementar estrategias de fidelización, beneficios exclusivos o programas de recompensas que permitan conservarlos y aumentar su recurrencia de compra.

Finalmente, el análisis por categoría, método de pago, tiempo y centro comercial permitió transformar los datos en información útil para la toma de decisiones. En conjunto, los resultados demuestran que la construcción de la bodega de datos y el proceso ETL facilitaron una mejor organización de la información y permitieron obtener hallazgos relevantes sobre el comportamiento de ventas. Esto confirma la importancia de integrar almacenamiento estructurado, consultas analíticas y visualización de datos como apoyo a los procesos de inteligencia de negocios.
