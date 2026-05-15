# Pipeline completo del proyecto: Angry Voice Recognition ESP32-S3 N16R8

## 1. Descripcion general

Este proyecto implementa un sistema embebido para detectar enojo en la voz usando un ESP32-S3 con PSRAM. El sistema captura audio en tiempo real mediante un microfono I2S, calcula caracteristicas MFCC, ejecuta una red neuronal convolucional con TensorFlow Lite Micro y muestra el resultado mediante una matriz LED MAX72XX y un buzzer.

A diferencia del proyecto de identificacion de hablantes, aqui no se grabo un dataset propio. Para el entrenamiento se utilizo un dataset existente: ESD, ubicado en `Dataset/ESD/Emotional_Speech_Dataset/`.

El problema se adapto a clasificacion binaria:

- `Angry`: voz con emocion de enojo.
- `Other`: cualquier otra emocion disponible en el dataset.

El sistema final no reconoce cinco emociones por separado. Su objetivo es responder una pregunta concreta: la voz detectada corresponde a enojo o no.

---

## 2. Dataset utilizado

El dataset usado es ESD, almacenado en:

```text
Dataset/ESD/Emotional_Speech_Dataset/
```

El dataset esta dividido por idioma:

```text
Chinese/
English/
```

Dentro de cada idioma se encuentran cinco emociones:

```text
angry
happy
neutral
sad
surprise
```

Cada emocion contiene tres subconjuntos:

```text
train
test
evaluation
```

La distribucion por cada emocion es:

| Subconjunto | Cantidad de audios |
|---|---:|
| train | 3000 |
| test | 300 |
| evaluation | 200 |

La estructura general es:

```text
Dataset/
└── ESD/
    └── Emotional_Speech_Dataset/
        ├── Chinese/
        │   ├── angry/
        │   │   ├── train/       -> 3000 audios
        │   │   ├── test/        -> 300 audios
        │   │   └── evaluation/  -> 200 audios
        │   ├── happy/
        │   ├── neutral/
        │   ├── sad/
        │   └── surprise/
        │
        └── English/
            ├── angry/
            │   ├── train/       -> 3000 audios
            │   ├── test/        -> 300 audios
            │   └── evaluation/  -> 200 audios
            ├── happy/
            ├── neutral/
            ├── sad/
            └── surprise/
```

---

## 3. Conversion de cinco emociones a clasificacion binaria

Aunque el dataset original contiene cinco emociones, el proyecto lo transforma en un problema binario.

| Emocion original | Clase final | Etiqueta |
|---|---|---:|
| angry | Angry | 1 |
| happy | Other | 0 |
| neutral | Other | 0 |
| sad | Other | 0 |
| surprise | Other | 0 |

En el notebook se usan dos clases:

```python
clases_vec = [0, 1]
emotion_vec = ["Other", "Angry"]
```

Esto significa que el modelo aprende a distinguir `Angry` contra todo lo demas.

---

## 4. Pipeline general

```text
Dataset ESD existente
        ↓
Leer idiomas: Chinese y English
        ↓
Leer emociones: angry, happy, neutral, sad, surprise
        ↓
Leer subconjuntos: train, test, evaluation
        ↓
Asignar etiquetas binarias:
        angry = Angry = 1
        happy/neutral/sad/surprise = Other = 0
        ↓
Cargar audio a 16 kHz y mono
        ↓
Detectar inicio aproximado del sonido
        ↓
Recortar 2 segundos de audio util
        ↓
Rellenar con ceros si el audio queda corto
        ↓
Agregar ruido blanco leve
        ↓
Calcular MFCC manualmente
        ↓
Recortar MFCC central a 63 frames
        ↓
Entrenar CNN binaria
        ↓
Evaluar con test y evaluation
        ↓
Exportar a TensorFlow Lite
        ↓
Convertir modelo a arreglo C/C++
        ↓
Integrar modelo en ESP32-S3
        ↓
Capturar audio por I2S
        ↓
Calcular MFCC en firmware
        ↓
Ejecutar inferencia con TensorFlow Lite Micro
        ↓
Mostrar Other o Angry en matriz LED
        ↓
Activar buzzer si se detecta Angry
```

---

# Parte 1: Proceso en el notebook

## 5. Carga del dataset

El notebook recorre la carpeta `Dataset/ESD/Emotional_Speech_Dataset/` y toma audios tanto de `English` como de `Chinese`.

Para cada idioma se recorren las emociones originales del dataset. La emocion `angry` se etiqueta como clase positiva y las demas se agrupan en `Other`.

Esto permite aprovechar un dataset existente sin grabar audios nuevos. La ventaja es que el dataset ya esta organizado, etiquetado y dividido en subconjuntos de entrenamiento, prueba y evaluacion.

---

## 6. Separacion train, test y evaluation

El dataset ya viene separado en tres subconjuntos:

- `train`: usado para entrenar la red neuronal.
- `test`: usado para probar el modelo con datos no vistos durante el entrenamiento.
- `evaluation`: usado como evaluacion adicional para medir generalizacion.

La estructura evita mezclar audios de entrenamiento con audios de prueba.

---

## 7. Formato de audio usado

El notebook procesa todos los audios con la misma configuracion:

```python
sampling_rate = 16000
duration = 2
num_muestras = sampling_rate * duration
```

Por lo tanto, cada muestra final tiene:

```text
16000 muestras/segundo x 2 segundos = 32000 muestras
```

Este formato coincide con el firmware, donde el ESP32 procesa bloques de 2 segundos.

---

## 8. Deteccion del inicio del sonido

El notebook no toma simplemente los primeros 2 segundos del audio. Primero busca el inicio aproximado de la voz.

Para esto calcula cambios de energia en el audio. Cuando detecta un cambio suficientemente grande, toma ese punto como inicio del segmento util.

La idea es evitar que el segmento contenga demasiado silencio al inicio.

Flujo:

```text
Audio original
        ↓
Analisis de energia
        ↓
Deteccion del inicio de voz
        ↓
Recorte desde ese punto
        ↓
Segmento final de 2 segundos
```

---

## 9. Igualacion de longitud

Todos los audios deben tener la misma longitud para poder entrar al modelo.

Si el audio es mayor a 2 segundos, se recorta.

Si el audio es menor a 2 segundos, se rellena con ceros.

```text
Audio corto -> se completa con ceros
Audio largo -> se recorta a 2 segundos
```

El resultado final siempre es:

```text
32000 muestras
```

---

## 10. Aumento de datos con ruido blanco

El notebook agrega ruido blanco leve a las senales de audio.

Este ruido no busca cambiar la emocion, sino ayudar a que el modelo sea mas robusto ante pequenas variaciones.

Conceptualmente:

```text
Audio limpio + ruido blanco leve = audio aumentado
```

Esto puede ayudar a que el modelo no dependa de condiciones perfectas de audio.

---

## 11. Extraccion de MFCC

Despues de preparar los audios, el notebook calcula caracteristicas MFCC de forma manual.

Los MFCC representan la informacion frecuencial de la voz y son utiles para tareas de reconocimiento de voz y emociones.

El proceso es:

```text
Audio de 2 segundos
        ↓
Preenfasis
        ↓
Ventana Hamming
        ↓
FFT
        ↓
Espectro de potencia
        ↓
Filtros Mel
        ↓
Logaritmo natural
        ↓
DCT
        ↓
Matriz MFCC
```

---

## 12. Parametros MFCC

El notebook y el firmware usan parametros equivalentes:

| Parametro | Valor |
|---|---:|
| Frecuencia de muestreo | 16000 Hz |
| Duracion | 2 segundos |
| Muestras por segmento | 32000 |
| Tamano de ventana | 512 |
| Salto entre ventanas | 256 |
| Frecuencia minima | 20 Hz |
| Frecuencia maxima | 8000 Hz |
| Bandas Mel | 20 |
| Frames finales | 63 |

En firmware estos parametros aparecen en `MFCC.h`:

```cpp
#define SAMPLING_RATE 16000
#define TOTAL_TIME 2
#define SHAPE_INPUT SAMPLING_RATE*TOTAL_TIME
#define SIZE_WINDOW 512
#define WINDOW_STEP 256
#define MEL_BANDS 20
```

---

## 13. Recorte central a 63 frames

El calculo MFCC completo produce mas frames temporales, pero el modelo fue disenado para recibir una entrada de:

```text
20 x 63 x 1
```

Por eso se recorta la parte central de la matriz MFCC.

La justificacion es que, despues de alinear el inicio de la voz, la informacion mas util suele estar en la zona central del segmento.

---

## 14. Forma final de entrada del modelo

Cada audio queda representado como una matriz:

```text
20 bandas Mel x 63 frames x 1 canal
```

La forma final es:

```text
20 x 63 x 1
```

Esta forma coincide con `NeuralNetwork.cpp`, donde el firmware configura la entrada del modelo como:

```cpp
input->dims->data[0] = 1;
input->dims->data[1] = 20;
input->dims->data[2] = 63;
input->dims->data[3] = 1;
```

---

# Parte 2: Entrenamiento del modelo

## 15. Modelo CNN binario

El notebook entrena una red neuronal convolucional ligera para clasificacion binaria.

La entrada del modelo es:

```text
20 x 63 x 1
```

La salida es una sola neurona con activacion sigmoid.

```text
Salida cercana a 0 -> Other
Salida cercana a 1 -> Angry
```

---

## 16. Arquitectura general

La arquitectura usada en el notebook es una CNN ligera inspirada en LightCNN.

Estructura general:

```text
Conv2D 32 filtros 3x3 + ReLU
MaxPooling2D
Conv2D 16 filtros 3x3 + ReLU
Conv2D 16 filtros 3x3 + ReLU
MaxPooling2D
Flatten
Dense 32 + ReLU
Dropout
Dense 32 + ReLU
Dropout
Dense 1 + Sigmoid
```

El modelo se compila con:

```text
Loss: binary_crossentropy
Optimizer: Adam
Metric: accuracy
```

La funcion `binary_crossentropy` es adecuada porque el problema tiene dos clases: `Other` y `Angry`.

---

## 17. Entrenamiento

El notebook entrena el modelo con los datos de `train` y valida su desempeno con datos separados.

Tambien guarda el mejor modelo mediante checkpoint, monitoreando la perdida de validacion.

Esto evita quedarse necesariamente con la ultima epoca, ya que la mejor version puede ocurrir antes de terminar el entrenamiento.

---

## 18. Evaluacion del modelo

El notebook evalua el modelo usando conjuntos que no forman parte del entrenamiento.

Se calculan metricas como:

- Accuracy
- Precision
- Recall
- F-score
- Specificity
- Matriz de confusion
- Curva ROC
- AUC

La matriz de confusion permite observar cuantos audios Angry fueron detectados correctamente y cuantos audios Other fueron confundidos como Angry.

---

## 19. Uso de umbrales

La salida del modelo es una probabilidad entre 0 y 1.

Ejemplo:

```text
0.15 -> probablemente Other
0.92 -> probablemente Angry
```

Durante pruebas se pueden usar diferentes umbrales. En el firmware se usa:

```cpp
int result = (val_angry >= 0.70) ? 1 : 0;
```

Esto significa que solo se declara `Angry` cuando la salida del modelo es al menos 0.70.

Este umbral ayuda a reducir falsos positivos.

---

## 20. Exportacion del modelo

Despues del entrenamiento, el modelo se exporta a TensorFlow Lite.

El notebook genera archivos como:

```text
ModelMFCC_CNN_Angry_16k_v2.tflite
NuevoModelo_Angry_16k_512_256_int8.tflite
model_data.cc
```

Posteriormente, el modelo se adapta al firmware como:

```text
model_data.h
```

Este archivo contiene el modelo como un arreglo de bytes en C/C++.

---

# Parte 3: Firmware ESP32-S3

## 21. Funcion general del firmware

El firmware realiza el proceso en tiempo real:

```text
Captura audio por I2S
        ↓
Normaliza y filtra la senal
        ↓
Calcula energia para descartar silencio
        ↓
Calcula MFCC
        ↓
Copia MFCC al tensor de entrada
        ↓
Ejecuta inferencia
        ↓
Interpreta salida sigmoid
        ↓
Muestra emocion en matriz LED
        ↓
Activa buzzer si detecta Angry
```

---

## 22. Configuracion de hardware

El firmware usa:

- ESP32-S3 N16R8
- PSRAM habilitada
- Microfono I2S
- Matriz LED MAX72XX
- Buzzer

Pines principales:

```text
Buzzer: GPIO 21
I2S BCLK: GPIO 2
I2S LRCL/WS: GPIO 7
I2S DATA IN: GPIO 1
Matriz LED MOSI: GPIO 13
Matriz LED SCK: GPIO 14
Matriz LED CS: GPIO 10
```

---

## 23. Configuracion I2S

El microfono se configura a:

```text
Frecuencia: 16000 Hz
Resolucion: 32 bits
Canal: derecho
Modo: I2S RX
DMA buffers: 8
DMA length: 512
```

Esto permite capturar audio digital directamente desde el microfono.

---

## 24. Normalizacion y filtrado en firmware

La funcion `audio_normalized()` convierte la senal cruda del I2S a una senal flotante adecuada para MFCC.

El proceso incluye:

- limpieza de bits de baja amplitud;
- desplazamiento de la muestra;
- preenfasis;
- suavizado;
- filtro pasa-altas;
- centrado;
- ganancia;
- limitacion entre -1.0 y 1.0.

Esto busca que la senal capturada por el ESP32 sea compatible con la distribucion usada durante el entrenamiento.

---

## 25. Filtro de energia

Antes de calcular MFCC, el firmware calcula la energia media del audio.

Si la energia es menor que el umbral, el sistema considera que hay silencio o ruido muy bajo.

En ese caso:

```text
No ejecuta el modelo
Muestra estado Other o sin sonido
Mantiene apagado el buzzer
Libera memoria
```

Este filtro reduce inferencias innecesarias.

---

## 26. Calculo de MFCC en firmware

El archivo `MFCC.cpp` implementa el mismo proceso usado en entrenamiento:

```text
Preenfasis
Ventana Hamming
FFT
Espectro de potencia
Filtros Mel
Logaritmo
DCT
Recorte central a 63 frames
```

Tambien incluye optimizaciones:

- ventana Hamming precalculada;
- matriz DCT precalculada;
- matrices grandes en PSRAM;
- buffers internos para reducir trafico de memoria.

---

## 27. Uso de PSRAM

El proyecto usa PSRAM porque el modelo y las matrices de audio consumen memoria considerable.

Se reserva memoria para:

- audio crudo;
- audio normalizado;
- matriz MFCC;
- tensor arena de TensorFlow Lite Micro.

En `platformio.ini` se habilita PSRAM:

```ini
board_build.psram = 1
-D BOARD_HAS_PSRAM
-D CONFIG_SPIRAM
-D CONFIG_SPIRAM_USE_MALLOC
```

En `NeuralNetwork.cpp`, el `tensor_arena` se reserva en PSRAM con un tamano aproximado de 600 KB.

---

## 28. TensorFlow Lite Micro

La clase `NeuralNetwork` se encarga de:

- verificar PSRAM;
- cargar el modelo desde `model_data.h`;
- verificar compatibilidad con `TFLITE_SCHEMA_VERSION`;
- registrar las operaciones necesarias;
- reservar el `tensor_arena`;
- crear el interprete;
- asignar tensores;
- ejecutar la inferencia.

Las operaciones registradas incluyen:

```text
FullyConnected
Logistic
Reshape
Conv2D
MaxPool2D
L2Normalization
```

Estas operaciones coinciden con una red convolucional binaria.

---

## 29. Interpretacion de salida

El modelo entrega un valor entre 0 y 1 asociado a la probabilidad de Angry.

En el firmware:

```cpp
float val_angry = outputTensor->data.f[0];
int result = (val_angry >= 0.70) ? 1 : 0;
```

Si `result` es 0:

```text
Other
```

Si `result` es 1:

```text
Angry
```

Tambien se imprime una barra visual en el monitor serial:

```text
ANGRY: 85% [#################---]
```

---

## 30. Salida visual y sonora

El proyecto usa una matriz LED MAX72XX para mostrar el resultado.

En `emotions.cpp` se definen patrones para:

- Other
- Angry


Cuando se detecta Angry, el buzzer se activa:

```cpp
digitalWrite(BUZZER_PIN, (feeling == 1) ? HIGH : LOW);
```

Esto permite que el sistema no solo clasifique internamente, sino que tambien comunique el resultado al usuario.

---

# Parte 4: Resumen para exposicion

## 31. Explicacion corta

Este proyecto detecta enojo en la voz usando un ESP32-S3 con PSRAM. Para entrenar el modelo se uso un dataset existente llamado ESD, ubicado en `Dataset/ESD/Emotional_Speech_Dataset/`. El dataset esta dividido en `Chinese` y `English`, y contiene las emociones `angry`, `happy`, `neutral`, `sad` y `surprise`. Cada emocion contiene 3000 audios de entrenamiento, 300 de prueba y 200 de evaluacion.

Aunque el dataset tiene cinco emociones, el proyecto lo adapta a clasificacion binaria. La emocion `angry` se etiqueta como `Angry`, mientras que `happy`, `neutral`, `sad` y `surprise` se agrupan como `Other`.

En el notebook se cargan los audios a 16 kHz, se detecta el inicio de la voz, se recortan segmentos de 2 segundos, se agrega ruido blanco leve y se calculan MFCC. Cada audio se convierte en una matriz de `20 x 63 x 1`, que se usa para entrenar una CNN ligera con salida sigmoid.

Despues, el modelo se convierte a TensorFlow Lite y se integra en el firmware del ESP32. En tiempo real, el ESP32 captura audio por I2S, calcula MFCC, ejecuta la red neuronal y decide si la voz es `Other` o `Angry`. Si detecta `Angry`, muestra el patron correspondiente en la matriz LED y activa el buzzer.

---

## 32. Diferencia con el proyecto de identificacion de hablantes

| Proyecto | Dataset | Objetivo | Modelo |
|---|---|---|---|
| MiniXi SpeakerID | Grabaciones propias | Identificar quien habla | Mini Xi-vector + MLP |
| Angry Voice Recognition | Dataset ESD existente | Detectar enojo en la voz | MFCC + CNN binaria |

La diferencia principal es que en este proyecto no se graban voces propias. Se aprovecha un dataset ya etiquetado y se adapta a una tarea binaria de deteccion de enojo.

---

## 33. Conclusion

El proyecto demuestra como llevar un modelo de reconocimiento emocional desde el entrenamiento en Python hasta la ejecucion en un microcontrolador.

La parte mas importante es mantener coherencia entre el notebook y el firmware:

- misma frecuencia de muestreo;
- misma duracion de audio;
- mismo tamano de ventana;
- mismo salto;
- mismas bandas Mel;
- mismo recorte a 63 frames;
- misma forma de entrada `20 x 63 x 1`.

Gracias a esto, el modelo entrenado puede ejecutarse en tiempo real en el ESP32-S3 usando TensorFlow Lite Micro, PSRAM, micrófono I2S, matriz LED y buzzer.
