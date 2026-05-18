# Compact Near Real-Time Anger Detection on ESP32-S3-N16R8

Sistema embebido compacto para detectar enojo en la voz usando analisis MFCC, una CNN ligera hibrida inspirada en LightCNN-9 y TensorFlow Lite Micro sobre un **ESP32-S3-N16R8**.

El objetivo del proyecto no es clasificar todas las emociones del dataset, sino resolver una tarea binaria:

```text
Angry = 1
Other = 0
```

Crear una solucion embebida implica trabajar con limites de memoria, computo y latencia. Por eso el sistema no usa una red neuronal grande sobre audio crudo. En su lugar, transforma la senal de voz a **MFCC** y ejecuta una **CNN compacta** directamente en el microcontrolador.

<p align="center">
<img src="./DiagramaCNN.png" alt="MFCC and lightweight CNN architecture on ESP32-S3-N16R8" width="560px">
</p>

## Overview

El sistema realiza el procesamiento completo de forma local:

- Captura audio con un microfono I2S.
- Calcula MFCC en el ESP32-S3.
- Ejecuta inferencia con TensorFlow Lite Micro.
- Clasifica la muestra como `Other` o `Angry`.
- Muestra el resultado en una matriz LED MAX72XX.
- Activa un buzzer cuando detecta enojo.

Rendimiento medido en el **ESP32-S3-N16R8**:

| Medicion | Valor aproximado |
|---|---:|
| MFCC | ~190 ms |
| Inferencia CNN | ~219 ms |
| Procesamiento total | ~409 ms |
| RAM interna libre | ~229 KB |
| PSRAM libre | ~7 MB |

## Hardware Target

El firmware esta configurado para un **ESP32-S3-N16R8**:

| Elemento | Descripcion |
|---|---|
| ESP32-S3 | Microcontrolador con soporte para audio digital e inferencia ligera |
| N16 | 16 MB de Flash |
| R8 | 8 MB de PSRAM |
| Microfono | I2S, 32 bits, canal derecho |
| Salida visual | Matriz LED MAX72XX |
| Salida sonora | Buzzer |

La PSRAM es importante para reservar el `tensor_arena`, procesar buffers de audio de 2 segundos y almacenar matrices intermedias de MFCC, FFT, filtros Mel y DCT.

## Dataset

El entrenamiento usa el **ESD (Emotional Speech Dataset)**:

```text
Dataset/ESD/Emotional_Speech_Dataset/
```

El dataset contiene dos idiomas:

- `Chinese`
- `English`

Y cinco emociones:

- `angry`
- `happy`
- `neutral`
- `sad`
- `surprise`

Para este proyecto se convierte a clasificacion binaria:

| Emocion original | Clase final | Etiqueta |
|---|---|---:|
| angry | Angry | 1 |
| happy | Other | 0 |
| neutral | Other | 0 |
| sad | Other | 0 |
| surprise | Other | 0 |

## Audio Analysis: MFCC

El modelo no recibe audio crudo. Primero se extraen **Mel-Frequency Cepstral Coefficients (MFCC)**, que representan la energia de la voz en bandas perceptualmente relevantes.

Flujo de extraccion:

1. Normalizacion de audio.
2. Preenfasis.
3. Division en frames.
4. Ventana Hamming.
5. FFT.
6. Espectro de potencia.
7. Banco de filtros Mel.
8. Transformacion logaritmica.
9. DCT.
10. Matriz MFCC.

Parametros principales:

| Parametro | Valor |
|---|---:|
| Frecuencia de muestreo | 16 kHz |
| Duracion por bloque | 2 s |
| Muestras por bloque | 32000 |
| Ventana | 512 muestras |
| Salto | 256 muestras |
| Bandas Mel | 20 |
| Frames MFCC completos | 124 |
| Entrada documentada del modelo | `20 x 124 x 1` |
| Region central usada en firmware | 63 frames |

Nota tecnica: el calculo completo genera `124` ventanas MFCC. El firmware conserva ese calculo y aplica un recorte central de `63 frames` como adaptacion embebida.

## Model: MFCC + Lightweight CNN

El modelo es una arquitectura **hibrida MFCC + CNN ligera** con caracteristicas inspiradas en **LightCNN-9**.

No debe describirse como un LightCNN-9 puro. La arquitectura real es mas pequena: usa **3 capas convolucionales principales**, dos etapas de `MaxPooling2D`, una capa densa de embedding y una salida sigmoid para clasificacion binaria.

### Architecture Overview

```text
Input MFCC: 20 x 124 x 1
        |
        v
Conv2D 32 filtros 3x3 + ReLU
        |
        v
MaxPooling2D 3x3
        |
        v
Conv2D 16 filtros 3x3 + ReLU
        |
        v
Conv2D 16 filtros 3x3 + ReLU
        |
        v
MaxPooling2D 3x3
        |
        v
Flatten
        |
        v
Dense 32 + ReLU
        |
        v
Dropout 0.36
        |
        v
Dense 32 + ReLU
        |
        v
Dropout 0.36
        |
        v
Dense 1 + Sigmoid
```

Resumen del modelo:

| Metrica | Valor |
|---|---:|
| Parametros totales | 21697 |
| FLOPS aproximados | 10039618 |
| Salida | Sigmoid binaria |
| Umbral en firmware | `0.70` |

La salida sigmoid produce un valor entre `0.0` y `1.0`. Si la probabilidad de enojo es mayor o igual a `0.70`, el firmware declara la clase `Angry`.

## Embedded Pipeline

```text
Dataset ESD
        |
        v
Etiquetado binario Angry / Other
        |
        v
Audio mono a 16 kHz
        |
        v
Segmentos de 2 segundos
        |
        v
MFCC 20 x 124 x 1
        |
        v
CNN ligera hibrida
        |
        v
Exportacion a TensorFlow Lite
        |
        v
Conversion a arreglo C/C++
        |
        v
TensorFlow Lite Micro en ESP32-S3-N16R8
        |
        v
Matriz LED + buzzer
```

## Firmware

Archivos principales:

| Archivo | Funcion |
|---|---|
| `src/main.cpp` | Captura I2S, energia, MFCC, inferencia y salida |
| `src/MFCC.cpp` | Implementacion de MFCC en ESP32 |
| `src/MFCC.h` | Parametros de audio y MFCC |
| `src/NeuralNetwork.cpp` | Carga del modelo, tensor arena e inferencia |
| `src/NeuralNetwork.h` | Clase de red neuronal |
| `src/model_data.h` | Declaracion del modelo TFLite |
| `src/modelo_tesis_binario_Angry_finalv5.cc` | Modelo convertido a bytes C/C++ |
| `src/emotions.cpp` | Patrones para matriz LED |

## Requirements

- PlatformIO.
- Framework Arduino para ESP32.
- Board configurada como `rymcu-esp32-s3-devkitc-1`.
- PSRAM habilitada.
- Libreria `MD_MAX72XX` incluida en `lib/`.
- TensorFlow Lite Micro incluido en `lib/tfmicro/`.

Compilacion:

```bash
pio run
```

Carga al ESP32-S3:

```bash
pio run -t upload
```

Monitor serial:

```bash
pio device monitor
```

## Documentation

- [Pipeline V2 del proyecto](PIPELINE_V2_ANGRY_VOICE_RECOGNITION.md): documentacion tecnica completa del pipeline, arquitectura, firmware, memoria y tiempos medidos.

## Summary

Este proyecto demuestra un pipeline completo de IA embebida: dataset, extraccion MFCC, entrenamiento de una CNN ligera, conversion a TensorFlow Lite y ejecucion local en un ESP32-S3-N16R8 con salida visual y sonora.
