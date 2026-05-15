# Angry Voice Recognition ESP32-S3 N16R8

Sistema embebido para detectar enojo en la voz usando ESP32-S3, PSRAM, MFCC, una CNN binaria y TensorFlow Lite Micro.

## Descripcion

Este proyecto captura audio en tiempo real mediante un microfono I2S, calcula caracteristicas MFCC y ejecuta un modelo de red neuronal convolucional para determinar si la voz corresponde a la emocion `Angry` o a la clase `Other`.

Cuando se detecta `Angry`, el sistema muestra el resultado en una matriz LED MAX72XX y activa un buzzer.

## Dataset

Para el entrenamiento no se grabaron audios propios. Se utilizo un dataset existente llamado ESD:

```text
Dataset/ESD/Emotional_Speech_Dataset/
```

El dataset esta dividido por idioma:

- `Chinese`
- `English`

Cada idioma contiene cinco emociones:

- `angry`
- `happy`
- `neutral`
- `sad`
- `surprise`

Cada emocion contiene:

| Subconjunto | Audios |
|---|---:|
| train | 3000 |
| test | 300 |
| evaluation | 200 |

## Clasificacion binaria

El dataset original contiene cinco emociones, pero el proyecto lo adapta a dos clases:

| Emocion original | Clase final | Etiqueta |
|---|---|---:|
| angry | Angry | 1 |
| happy | Other | 0 |
| neutral | Other | 0 |
| sad | Other | 0 |
| surprise | Other | 0 |

## Pipeline

```text
Dataset ESD existente
        ↓
Leer Chinese y English
        ↓
Leer angry, happy, neutral, sad, surprise
        ↓
Convertir etiquetas a Angry / Other
        ↓
Cargar audio a 16 kHz y mono
        ↓
Detectar inicio del sonido
        ↓
Recortar 2 segundos
        ↓
Agregar ruido blanco leve
        ↓
Calcular MFCC
        ↓
Recortar a 20 x 63 x 1
        ↓
Entrenar CNN binaria
        ↓
Exportar a TensorFlow Lite
        ↓
Integrar en ESP32-S3
        ↓
Inferencia en tiempo real
```

## Caracteristicas tecnicas

- Microcontrolador: ESP32-S3 N16R8
- PSRAM habilitada
- Microfono I2S
- Frecuencia de muestreo: 16 kHz
- Duracion de audio: 2 segundos
- Muestras por bloque: 32000
- Ventana MFCC: 512
- Salto: 256
- Bandas Mel: 20
- Frames finales: 63
- Entrada del modelo: `20 x 63 x 1`
- Salida del modelo: sigmoid binaria
- Clases: `Other` y `Angry`

## Firmware

Archivos principales:

```text
src/main.cpp          -> flujo principal de captura, MFCC e inferencia
src/MFCC.cpp          -> calculo de MFCC en ESP32
src/MFCC.h            -> parametros de audio y MFCC
src/NeuralNetwork.cpp -> carga e inferencia con TensorFlow Lite Micro
src/NeuralNetwork.h   -> clase de red neuronal
src/emotions.cpp      -> patrones para matriz LED
src/emotions.h        -> prototipos de salida visual
src/model_data.h      -> modelo convertido a arreglo C/C++
```

## Salida del sistema

El sistema puede mostrar:

- `Other`: voz sin enojo detectado.
- `Angry`: voz con enojo detectado.
- `No Sound`: ausencia de sonido o energia insuficiente.

Si se detecta `Angry`, el buzzer se activa.

## Documentacion
## Documentación

- [Pipeline completo del proyecto](PIPELINE_COMPLETO_ANGRY_VOICE_RECOGNITION.md) : pipeline completo del proyecto.
- [Cambios de documentación](cambios_documentacion_angry_voice.txt) : resumen de cambios/documentacion.

## Resumen

Este proyecto demuestra la integracion de procesamiento de audio, aprendizaje automatico y sistemas embebidos. El modelo se entrena en Python con el dataset ESD y se ejecuta en tiempo real en ESP32-S3 usando TensorFlow Lite Micro.
