---
title: "Práctica 6"
date: 2025-09-09
---

# Práctica 6

## Contexto
En esta práctica número 6 del curso tenemos la siguiente problemática: Los centros comerciales buscan conocer mejor a sus clientes, pero el problema a resolver es la segmentación de clientes, es decir, agruparlos según sus características y comportamientos para poder diseñar campañas de marketing más personalizadas, ofrecer promociones específicas, invertir mejor en publicidad y comprender cómo compran los distintos tipos de clientes.

## Objetivos
- Identificar 3-5 segmentos de clientes distintos usando K-Means.
- Aplicar técnicas de normalización (MinMax, Standard, Robust).
- Usar PCA para reducción de dimensionalidad y visualización.
- Comparar PCA con métodos de selección de features.
- Interpretar resultados desde perspectiva de negocio.

## Actividades (con tiempos estimados)
- Parte 1 (40min)

## Desarrollo


## Evidencias
- Se adjunta imagen "resultado-t6-parte1.1.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.2.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.3.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.4.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.5.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.6.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.7.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.8.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.9.png" en `docs/assets/`
- Se adjunta imagen "resultado-t6-parte1.10.png" en `docs/assets/`

## Reflexión

---

# Machine Learning Clásico: Clustering y PCA - Mall Customer Segmentation

## Setup inicial: Código

```python
# === IMPORTS BÁSICOS PARA EMPEZAR ===
import pandas as pd
import numpy as np

print("Iniciando análisis de Mall Customer Segmentation Dataset")
print("Pandas y NumPy cargados - listos para trabajar con datos")
```

## Parte 1: Descripción
Aquí empezamos por cargar el dataset de clientes, analizaremos sus atributos, los tipos de datos que manejamos y más.

## Parte 1: Código

```python
# Descargar desde GitHub (opción más confiable)
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"

df_customers = pd.read_csv(url)

print("INFORMACIÓN DEL DATASET:")
print(f"Shape: {df_customers.shape[0]} filas, {df_customers.shape[1]} columnas")
print(f"Columnas: {list(df_customers.columns)}")
print(f"Memoria: {df_customers.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\nPRIMERAS 5 FILAS:")
df_customers.head()

# === ANÁLISIS DE TIPOS Y ESTRUCTURA ===
print("INFORMACIÓN DETALLADA DE COLUMNAS:")
print(df_customers.info())

print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
df_customers.describe()

# === ANÁLISIS DE GÉNERO ===
print("DISTRIBUCIÓN POR GÉNERO:")
gender_counts = df_customers['Genre'].value_counts()
print(gender_counts)
print(f"\nPorcentajes:")
for gender, count in gender_counts.items():
    pct = (count / len(df_customers) * 100)
    print(f"   {gender}: {pct:.1f}%")

# === ESTADÍSTICAS DE VARIABLES DE SEGMENTACIÓN ===
numeric_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

print("ESTADÍSTICAS CLAVE:")
print(df_customers[numeric_vars].describe().round(2))

print(f"\nRANGOS OBSERVADOS:")
for var in numeric_vars:
    min_val, max_val = df_customers[var].min(), df_customers[var].max()
    mean_val = df_customers[var].mean()
    print(f"   {var}: {min_val:.0f} - {max_val:.0f} (promedio: {mean_val:.1f})")

# === DETECCIÓN DE OUTLIERS USANDO IQR ===
print("DETECCIÓN DE OUTLIERS:")

outlier_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in outlier_cols:
    Q1 = df_customers[col].quantile(0.25)
    Q3 = df_customers[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calcular límites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Encontrar outliers
    outliers = df_customers[(df_customers[col] < lower_bound) | 
                           (df_customers[col] > upper_bound)]

    print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df_customers)*100:.1f}%)")
    print(f"      Límites normales: {lower_bound:.1f} - {upper_bound:.1f}")

```
#### Resultados: info del dataset
![Tabla comparativa](../assets/resultado-t6-parte1.1.png)

Cargamos el dataset usando la funcion read.csv() de pandas y vemos información general de las columnas que manipularemos, cantidad de columnas, cantidad de filas, memoria, tipos de datos y memoria usada por el dataset.

#### Resultados: análisis de datos
![Tabla comparativa](../assets/resultado-t6-parte1.2.png)

Aqui analizaremos estadísticas claves del dataset, como la cantidad de mujeres y hombres, vemos métricas claves en los atributos numéricos del dataset, por ejemplo, la media de edad, de ingresos, el minimo y máximo de edad, y también tenemos los valores de estos atributos para los quartiles, Q1, Q2 y Q3, representando el 25%, 50% y 75% de la información respectivamente. Posteriormente, observamos los rangos que manejamos en nuestros datos, en este caso la edad, el ingreso anual y el puntaje de comprador; para lograr esto usamos las funciones de min(), max() y mean(), de manera de obtener el mínimo, máximo y promedio, facilitando el cálculo.
Finalmente utilizamos los quartiles que nombramos para calcular outlines de los atributos, es decir, vemos el límite inferior y superior en el que se varia esa métrica.

```python
# === IMPORTS PARA VISUALIZACIÓN ===
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

# === HISTOGRAMAS DE VARIABLES PRINCIPALES ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribuciones de Variables Clave', fontsize=14, fontweight='bold')

vars_to_plot = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (var, color) in enumerate(zip(vars_to_plot, colors)):
    axes[i].hist(df_customers[var], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frecuencia')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Resultados: visualización de datos
![Tabla comparativa](../assets/resultado-t6-parte1.3.png)

Aqui observamos la frecuencia de los datos, para lograr esto, usamos la libreria matplotlib que nos permite graficar y la seaborn para agregarle formato a la visualización.

```python
# === SCATTER PLOTS PARA RELACIONES CLAVE ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Relaciones Entre Variables', fontsize=14, fontweight='bold')

# Age vs Income
axes[0].scatter(df_customers['Age'], df_customers['Annual Income (k$)'], 
                alpha=0.6, color='#96CEB4', s=50)
axes[0].set_xlabel('Age (años)')
axes[0].set_ylabel('Annual Income (k$)')
axes[0].set_title('Age vs Income')
axes[0].grid(True, alpha=0.3)

# Income vs Spending Score ⭐ CLAVE PARA SEGMENTACIÓN
axes[1].scatter(df_customers['Annual Income (k$)'], df_customers['Spending Score (1-100)'], 
                alpha=0.6, color='#FFEAA7', s=50)
axes[1].set_xlabel('Annual Income (k$)')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].set_title('Income vs Spending Score (CLAVE)')
axes[1].grid(True, alpha=0.3)

# Age vs Spending Score
axes[2].scatter(df_customers['Age'], df_customers['Spending Score (1-100)'], 
                alpha=0.6, color='#DDA0DD', s=50)
axes[2].set_xlabel('Age (años)')
axes[2].set_ylabel('Spending Score (1-100)')
axes[2].set_title('Age vs Spending Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Resultados: relación entre atributos
![Tabla comparativa](../assets/resultado-t6-parte1.4.png)

En esta visualización buscamos observar la relación entre los atributos, por ejemplo, el puntaje de compra según la edad o el ingreso anual segun la edad/puntaje de compra.

```python
# === MATRIZ DE CORRELACIÓN ===
correlation_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
corr_matrix = df_customers[correlation_vars].corr()

print("MATRIZ DE CORRELACIÓN:")
print(corr_matrix.round(3))

# Visualizar matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            fmt='.3f', linewidths=0.5, square=True)
plt.title('Matriz de Correlación - Mall Customers')
plt.tight_layout()
plt.show()

print(f"\nCORRELACIÓN MÁS FUERTE:")
# Encontrar la correlación más alta (excluyendo diagonal)
corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_corr = corr_flat.stack().idxmax()
max_val = corr_flat.stack().max()
print(f"   {max_corr[0]} ↔ {max_corr[1]}: {max_val:.3f}")
```
#### Resultados: matriz de correlación
![Tabla comparativa](../assets/resultado-t6-parte1.6.png)

En esta matriz de correlación buscamos analizar entre atributos, cual son los que influyen más entre sí. Por ejemplo, se observa que la edad no influye tanto en el puntaje de compra, si no que el ingreso anual es el que tiene más peso en ese aspecto.

```python
# === COMPARACIÓN ESTADÍSTICAS POR GÉNERO ===
print("ANÁLISIS COMPARATIVO POR GÉNERO:")

gender_stats = df_customers.groupby('Genre')[numeric_vars].agg(['mean', 'std']).round(2)
print(gender_stats)

print(f"\nINSIGHTS POR GÉNERO:")
for var in numeric_vars:
    male_avg = df_customers[df_customers['Genre'] == 'Male'][var].mean()
    female_avg = df_customers[df_customers['Genre'] == 'Female'][var].mean()

    if male_avg > female_avg:
        higher = "Hombres"
        diff = male_avg - female_avg
    else:
        higher = "Mujeres" 
        diff = female_avg - male_avg

print(f"   {var}: {higher} tienen promedio más alto (diferencia: {diff:.1f})")
```
#### Resultados: análisis del género
![Tabla comparativa](../assets/resultado-t6-parte1.7.png)

Observamos un pequeño análisis de la influencia que tiene el género en el el puntaje de compra, y en este caso las mujeres tienen un promedio superior al hombre, es decir, son clientes más redituables económicamente. Utilizamos funciones como la media, mean() para el análisis.

```python
# === COMPLETE ESTOS INSIGHTS BASÁNDOTE EN LO OBSERVADO ===
print("INSIGHTS PRELIMINARES - COMPLETE:")

print(f"\nCOMPLETE BASÁNDOTE EN TUS OBSERVACIONES:")
print(f"   Variable con mayor variabilidad: Ingreso anual (Annual Income (k$))")
print(f"   ¿Existe correlación fuerte entre alguna variable? Si, entre el ingreso anual y el puntaje de compra.")
print(f"   ¿Qué variable tiene más outliers? La edad.") 
print(f"   ¿Los hombres y mujeres tienen patrones diferentes? Si, las mujeres tienen tendencia a tener mayor puntaje de compra, es decir, comprar más.")
print(f"   ¿Qué insight es más relevante para el análisis? La relación entre ingresos y puntaje de compra, porque define segmentos de clientes valiosos.")
print(f"   ¿Qué 2 variables serán más importantes para clustering? Annual Income (k$) y Spending Score (1-100).")

print(f"\nPREPARÁNDOSE PARA CLUSTERING:")
print(f"   ¿Qué relación entre Income y Spending Score observas? La relación es que con mayor ingreso anual, mayor es el puntaje de compra.")
print(f"   ¿Puedes imaginar grupos naturales de clientes? Si, clientes de bajo ingreso/bajo gasto, alto ingreso/alto gasto, y un grupo intermedio.")
```

```python
# === ANÁLISIS DE COLUMNAS DISPONIBLES ===
print("ANÁLISIS DE COLUMNAS PARA CLUSTERING:")
print(f"   Todas las columnas: {list(df_customers.columns)}")
print(f"   Numéricas: {df_customers.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"   Categóricas: {df_customers.select_dtypes(include=[object]).columns.tolist()}")

# Identificar qué excluir y qué incluir
exclude_columns = ['CustomerID']  # ID no aporta información
numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_columns = ['Genre']

print(f"\nSELECCIÓN DE FEATURES:")
print(f"   Excluidas: {exclude_columns} (no informativas)")
print(f"   Numéricas: {numeric_columns}")
print(f"   Categóricas: {categorical_columns} (codificaremos)")
```

Aqui mostramos las columnas disponibles, y las clasificamos entre numéricas y categóricas, y a su vez excluimos las que no nos brindan información de la persona en cuestión.

```python
# === IMPORT ONEHOTENCODER ===
from sklearn.preprocessing import OneHotEncoder

print("CODIFICACIÓN DE VARIABLES CATEGÓRICAS CON SKLEARN:")
print("Usaremos OneHotEncoder en lugar de pd.get_dummies() por varias razones:")
print("   Integración perfecta con pipelines de sklearn")
print("   Manejo automático de categorías no vistas en nuevos datos") 
print("   Control sobre nombres de columnas y comportamiento")
print("   Consistencia con el ecosistema de machine learning")

# Crear y configurar OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Ajustar y transformar Genre
genre_data = df_customers[['Genre']]  # Debe ser 2D para sklearn
genre_encoded_array = encoder.fit_transform(genre_data)  # Método para ajustar y transformar

# Obtener nombres de las nuevas columnas
feature_names = encoder.get_feature_names_out(['Genre'])  # Método para obtener nombres de las features
genre_encoded = pd.DataFrame(genre_encoded_array, columns=feature_names)

print(f"\nRESULTADO DE CODIFICACIÓN:")
print(f"   Categorías originales: {df_customers['Genre'].unique()}")
print(f"   Columnas generadas: {list(genre_encoded.columns)}")
print(f"   Shape: {genre_data.shape} → {genre_encoded.shape}")

# Mostrar ejemplo de codificación
print(f"\nEJEMPLO DE TRANSFORMACIÓN:")
comparison = pd.concat([
    df_customers['Genre'].head().reset_index(drop=True),
    genre_encoded.head()
], axis=1)
print(comparison)
```

#### Resultados: OneHotEncoder
![Tabla comparativa](../assets/resultado-t6-parte1.8.png)

En esta seccion lo que buscamos es favorecer el modelo y para ello, creamos 2 categorias más a partir de los datos en sí, Male/Female ahora tienen sus atributos Genre_Female y Genre_Male, las cuales son booleans indicando si son hombres o mujeres.

```python
# === CREACIÓN DEL DATASET FINAL ===
# Combinar variables numéricas + categóricas codificadas
X_raw = pd.concat([
    df_customers[numeric_columns],
    genre_encoded
], axis=1)

print("DATASET FINAL PARA CLUSTERING:")
print(f"   Shape: {X_raw.shape}")
print(f"   Columnas: {list(X_raw.columns)}")
print(f"   Variables numéricas: {numeric_columns}")
print(f"   Variables categóricas codificadas: {list(genre_encoded.columns)}")
print(f"   Total features: {X_raw.shape[1]} (3 numéricas + 2 categóricas binarias)")
print(f"   Memoria: {X_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")

# === VERIFICACIONES ANTES DE CONTINUAR ===
print("VERIFICACIÓN DE CALIDAD:")

# 1. Datos faltantes
missing_data = X_raw.isnull().sum()
print(f"\nDATOS FALTANTES:")
if missing_data.sum() == 0:
    print("   PERFECTO! No hay datos faltantes")
else:
    for col, missing in missing_data.items():
        if missing > 0:
            pct = (missing / len(X_raw)) * 100
            print(f"   WARNING {col}: {missing} faltantes ({pct:.1f}%)")

# 2. Vista previa
print(f"\nVISTA PREVIA DEL DATASET:")
print(X_raw.head())

# 3. Tipos de datos
print(f"\nTIPOS DE DATOS:")
print(X_raw.dtypes)
```

#### Resultados: Nuevo DataSet
![Tabla comparativa](../assets/resultado-t6-parte1.9.png)

Creamos el dataset final con las variables creadas y numéricas que ya teniamos, y además, hacemos un breve chequeo si tenemos que pulir el dataset en caso de nulos o información faltante.

```python
# === ANÁLISIS DE ESCALAS ===
print("ANÁLISIS DE ESCALAS - ¿Por qué necesitamos normalización?")

print(f"\nESTADÍSTICAS POR VARIABLE:")
for col in X_raw.columns:
    if X_raw[col].dtype in ['int64', 'float64']:  # Solo numéricas
        min_val = X_raw[col].min()
        max_val = X_raw[col].max()
        mean_val = X_raw[col].mean()
        std_val = X_raw[col].std()

        print(f"\n   {col}:")
        print(f"      Rango: {min_val:.1f} - {max_val:.1f}")
        print(f"      Media: {mean_val:.1f}")
        print(f"      Desviación: {std_val:.1f}")

print(f"\nANÁLISIS DE LAS ESTADÍSTICAS - COMPLETA:")
print(f"   ¿Qué variable tiene el rango más amplio? Annual Income (k$).")
print(f"   ¿Cuál es la distribución de género en el dataset? Balanceada, un poquito más de mujeres.")
print(f"   ¿Qué variable muestra mayor variabilidad (std)? Annual Income (k$).")
print(f"   ¿Los clientes son jóvenes o mayores en promedio? Jóvenes-adultos, la edad promedio es de alrededor de 37 años.")
print(f"   ¿El income promedio sugiere qué clase social? Clase media.")
print(f"   ¿Por qué la normalización será crítica aca? Porque las variables están en escalas muy distintas, y puede llegar a sesgar los algoritmos de clustering.")

# Guardar para próximas fases
feature_columns = list(X_raw.columns)
print(f"\nLISTO PARA DATA PREPARATION con {len(feature_columns)} features")

```

#### Resultados: análisis de estadísticas
![Tabla comparativa](../assets/resultado-t6-parte1.10.png)

## Parte 2: Descripción
En esta parte buscamos iniciar la normalización del dataset.

## Parte 2: Código
```python


```

#### Resultados: análisis de estadísticas
![Tabla comparativa](../assets/resultado-t6-parte2.1.png)



