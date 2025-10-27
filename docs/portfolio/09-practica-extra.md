---
title: "Práctica 9 Extra"
date: 2025-09-30
---

# Práctica 9 Extra
## 📚 CNNs y Transfer Learning: Probamos nuevos datasets

## Contexto
En esta práctica extra extendemos el trabajo realizado en la práctica anterior con CNNs y Transfer Learning, pero aplicándolo a tres nuevos datasets de imágenes más variados y complejos. 

- **PlantVillage** 🌱 (clasificación de enfermedades en plantas)  
- **Cats vs Dogs** 🐱🐶 (clasificación binaria de animales)  
- **Food-101** 🍔 (clasificación multiclase de alimentos)

## Objetivos
- Reforzar el uso de **redes convolucionales (CNNs)** en diferentes contextos de visión computacional.  
- Aplicar **Transfer Learning** con modelos preentrenados como MobileNetV2.  
- Analizar cómo el rendimiento cambia entre datasets **simples, binarios y multiclase**.

## Actividades (con tiempos estimados)
- Parte 1 (min)
- 

## Desarrollo
Para cada dataset se repitió la misma estructura base:

1. **Carga y preprocesamiento de imágenes** usando `ImageDataGenerator`, con normalización y separación entre entrenamiento, validación y prueba.  
2. **Modelo CNN simple**, construido desde cero con dos capas convolucionales y una densa final.  
3. **Modelo con Transfer Learning** basado en MobileNetV2 (preentrenada en ImageNet), congelando las capas base y ajustando solo la parte superior.  
4. **Entrenamiento por 10 epochs** con `Adam` y `categorical_crossentropy` (o `binary_crossentropy` en el caso binario).  
5. **Evaluación final** sobre el conjunto de prueba y comparación de precisión entre ambos enfoques.


## Evidencias
- Se adjunta imagen "" en `docs/assets/`

## Reflexión


---
# CNNs y Transfer Learning con TensorFlow/Keras
## Setup inicial
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("🔧 GPU configurada correctamente")
else:
    print("🔧 Usando CPU")

tf.random.set_seed(42)
np.random.seed(42)

print("✅ Entorno TensorFlow/Keras configurado correctamente\n")
```

## Dataset 1 – PlantVillage 🌱
```python
print("📦 Descargando dataset PlantVillage desde Kaggle...")
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("✅ Dataset descargado en:", path, "\n")

# === CARGA Y PREPROCESAMIENTO ===
dataset_dir = os.path.join(path, "color") if os.path.exists(os.path.join(path, "color")) else path

train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names
print(f"📂 Clases detectadas: {len(class_names)} clases\n")

# === OPTIMIZACIÓN Y NORMALIZACIÓN ===
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
```

#### Resultados
<!-- ![Tabla comparativa](../assets/resultado-t9-extra-1.png) -->
