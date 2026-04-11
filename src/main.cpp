#include <Arduino.h>
#include <MD_MAX72xx.h>
#include "driver/i2s.h"
#include "soc/i2s_reg.h"
#include <esp_wifi.h>
#include <esp_bt.h>
#include "SD.h"
#include "SPI.h"
#include "MFCC.h"
#include "emotions.h"
#include "esp32-hal-timer.h"
#include "NeuralNetwork.h"
#include "tensorflow/lite/c/common.h" 

// Variables para mantener el estado del filtro de audio
float last_x = 0;
float last_y = 0;
// Contador de iteraciones para impresión 
int i = 0;

// --- DEFINICIÓN DE PINES ---

#define BUZZER_PIN 21 // Pin para el buzzer
#define MOSI_PIN   13 
#define SCK_PIN    14
#define CS_PIN     10
#define NUM_MAT    1 

// Pines I2S micro 
#define I2S_LRCL   7
#define I2S_DOUT   1
#define I2S_BCLK   2

#define BUFLEN SHAPE_INPUT

static const i2s_port_t i2s_num = I2S_NUM_0;
size_t number_bytes_read;
NeuralNetwork *nn = nullptr;

// Objeto para la matriz de LEDs
MD_MAX72XX mxObj = MD_MAX72XX(MD_MAX72XX::GENERIC_HW, MOSI_PIN, SCK_PIN, CS_PIN, NUM_MAT);
int Mat[1] = {0};
String emotion_vec[] = {"Other","Angry"};

// Configuracion I2S 
static const i2s_config_t i2s_config = {
  .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
  .sample_rate = 16000,  // Frecuencia de muestreo de 16 kHz para que coincida con el entrenamiento de la red neuronal
  .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
  .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
  .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S), 
  .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
  .dma_buf_count = 8, //64,
  .dma_buf_len = 512,
  .use_apll = false
};

// Configuración de pines I2S (sph0645)
static const i2s_pin_config_t pin_config = {
  .bck_io_num = I2S_BCLK,
  .ws_io_num = I2S_LRCL,
  .data_out_num = I2S_PIN_NO_CHANGE,
  .data_in_num = I2S_DOUT
};

// prototipos de mis funciones
void audio_normalized(int32_t *raw_signal, float *normalized_signal);
int interprets_output(TfLiteTensor *outputTensor);
float** allocate_mfcc_matrix_psram(int rows, int cols);
void free_mfcc_matrix_psram(float **matrix, int rows);
float* allocate_float_array_psram(size_t size);
void free_float_array_psram(float *arr);
int32_t* allocate_int32_array_psram(size_t size);
void free_int32_array_psram(int32_t *arr);

void setup() { 
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n===================================");
  Serial.println("Iniciando ESP32-S3 con PSRAM...");
  Serial.println("===================================");

  esp_bt_controller_disable();
  esp_wifi_stop();

  pinMode(BUZZER_PIN, OUTPUT);// Configuramos el pin del buzzer como salida
  digitalWrite(BUZZER_PIN, LOW);// Aseguramos que el buzzer esté apagado al inicio

  nn = new NeuralNetwork();
  if (!nn->begin()) {
    while (1) {
      digitalWrite(BUZZER_PIN, HIGH); delay(100); 
      digitalWrite(BUZZER_PIN, LOW); delay(100);  
    }
  }

  mxObj.begin();
  mxObj.clear();
  mxObj.control(MD_MAX72XX::INTENSITY, 1);
  emotion(0, Mat, 0, mxObj);

  i2s_driver_install(i2s_num, &i2s_config, 0, NULL);
  i2s_set_pin(i2s_num, &pin_config);
  i2s_set_clk(i2s_num, 16000, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
  i2s_zero_dma_buffer(i2s_num);
  
  Serial.println("(Sistemas listos)");
}

void loop() {
  digitalWrite(BUZZER_PIN, LOW); // Forzar apagado al iniciar cada ciclo
  int32_t *raw_signal = allocate_int32_array_psram(BUFLEN);
  if (!raw_signal) return;
  
  esp_err_t err = i2s_read(i2s_num, raw_signal, BUFLEN * sizeof(int32_t), &number_bytes_read, portMAX_DELAY);
  
  if (err == ESP_OK && number_bytes_read >= BUFLEN * sizeof(int32_t)) {
    float *inputAudio = allocate_float_array_psram(BUFLEN);
    if (!inputAudio) { free_int32_array_psram(raw_signal); return; }

    audio_normalized(raw_signal, inputAudio);
    free_int32_array_psram(raw_signal);

    float energia = 0;
    for (int j = 0; j < BUFLEN; j++) energia += abs(inputAudio[j]);
    energia /= BUFLEN;

    // Filtro básico de energía que permite ignorar segmentos de silencio o ruido muy bajo. El umbral de 0.020 es empírico y puede ajustarse según el entorno.
    if (energia < 0.016) {    // estaba a 0.02  vamos a probar con uno nuevo
        if (i % 20 == 0) Serial.printf("Silencio detectado (E: %.4f)\n", energia); 
        emotion(0, Mat, 0, mxObj);
        digitalWrite(BUZZER_PIN, LOW); 
        free_float_array_psram(inputAudio);
        return; 
    }

    float **mfcc_mat = allocate_mfcc_matrix_psram(MEL_BANDS, NUMBER_OF_WINDOWS);
    if (!mfcc_mat) { free_float_array_psram(inputAudio); return; }

    long int t1 = micros();
    mfccs(inputAudio, mfcc_mat);
    long int t2 = micros();
    free_float_array_psram(inputAudio);

    float *inputBuffer = nn->getInputBuffer();
    if (inputBuffer) {
      int index = 0;
      for (int r = 0; r < MEL_BANDS; r++) {
        for (int c = 0; c < NUMBER_OF_WINDOWS; c++) {
          inputBuffer[index++] = mfcc_mat[r][c];
        }
      }
    }// --- FIN DE PREPARACIÓN DE DATOS PARA LA RED NEURONAL ---

    // Liberar la matriz de MFCCs en PSRAM
    free_mfcc_matrix_psram(mfcc_mat, MEL_BANDS);

    long int t3 = micros();  // Tiempo antes de la inferencia

    bool success = nn->predict(); // Realizar la inferencia del modelo  ---------------------------------

    long int t4 = micros();// Tiempo después de la inferencia

    if (success) {
      TfLiteTensor *outputTensor = nn->getOutputTensor();  // salida del tensor---------------------
      
      Serial.printf("\n------------------- Iteración: %d ----------------\n", i++);
      
      int feeling = interprets_output(outputTensor);
      emotion(feeling, Mat, 0, mxObj);
      digitalWrite(BUZZER_PIN, (feeling == 1) ? HIGH : LOW);delay(200); // buzzer suena con angry
      //digitalWrite(LED_PLACA, (feeling == 1) ? HIGH : LOW); // LED de la placa se enciende con angry

      // --- SALIDA DETALLADA ESTILO ANTERIOR ---
      Serial.printf("Energía: %.4f\n", energia);   //energia de la señal (ruidos bajos deberían tener energía baja)
      Serial.printf("\nTiempo MFCC: %.2f ms\n", (t2 - t1) / 1000.0);   //tiempo que tarda el cálculo de los MFCCs entre 1000 para convertir
      Serial.printf("Tiempo Inferencia: %.2f ms\n", (t4 - t3) / 1000.0);  //tiempo que tarda la red neuronal en procesar los datos y dar un resultado
      Serial.printf("Tiempo Total: %.2f ms\n", (t4 - t1) / 1000.0);  //tiempo total desde que se empieza a procesar el audio hasta que se obtiene la predicción de la emoción
      
      // Información de memoria
      size_t freeRAM = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
      size_t freePSRAM = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
      Serial.printf("RAM Libre: %d KB | PSRAM Libre: %d KB\n", freeRAM / 1024, freePSRAM / 1024);  // Esto es útil para monitorear el uso de memoria 
      
      Serial.println("--------------------------------------------------\n");
    }
  } else {
    free_int32_array_psram(raw_signal);
  }
}//fin loop

// IMPLEMENTACIÓN DE FUNCIONES

// Función para normalizar y filtrar la señal de audio
void audio_normalized(int32_t *raw_signal, float *normalized_signal) {
    float sum = 0;
    float pre_emphasis_coeff = 0.97f; // Estándar para resaltar agudos en voz
    float prev_sample = 0;
    
    for(int j = 0; j < BUFLEN; j++) {
        raw_signal[j] &= 0xFFFFFF00;  //me ayuda a eliminar ruidos de baja amplitud
        float current_raw = (float)(raw_signal[j] >> 14);  //probar con 14
        
        // --- 1. PRE-ÉNFASIS --- ES PARA RESALTAR LOS AGUDOS DE LA VOZ, QUE SON CLAVE PARA DETECTAR EMOCIONES.
        float high_boost = current_raw - (pre_emphasis_coeff * prev_sample);
        prev_sample = current_raw;

        // --- 2. FILTRADO Y SUAVIZADO ---
        if (j > 1) {
            high_boost = (high_boost + normalized_signal[j-1] + normalized_signal[j-2]) / 3.0f;
        }

        // Filtro High-pass (Mantenemos 0.80 para ignorar ruidos agudos y constantes)
        //0.88 para ignorar silbidos, 0.80 para ignorar ruidos de bolsas
        float y = high_boost - last_x + (0.88f * last_y);   //esta operacion es la implementación de un filtro de paso alto que ayuda a eliminar ruidos constantes y resaltar cambios en la señal, lo cual es útil para detectar emociones en la voz. El coeficiente de 0.88f controla la cantidad de filtrado, permitiendo ignorar ruidos agudos y constantes como silbidos o ruidos de bolsas.
        last_x = high_boost;
        last_y = y;

        normalized_signal[j] = y; 
        sum += y;
    }

    float mean = sum / BUFLEN;

    // --- 3. GANANCIA DE COMPENSACIÓN --- 
    for(int j = 0; j < BUFLEN; j++) {
        float centered = normalized_signal[j] - mean;
        // Subimos un poco más la ganancia para compensar el pre-énfasis probamos con 25.0f para que los valores de los MFCCs sean adecuados para la red neuronal, ajustando la escala de la señal para que oscile aproximadamente entre -1.0 y 1.0, con un énfasis en las frecuencias relevantes para la detección de emociones en la voz.
        float val = (centered / 120000.0f) * 25.0f;   //100000.0f  60000.0f   estos son factores de ganancia que ajustan la escala de la señal para que los valores de los MFCCs sean adecuados para la red neuronal. Se pueden ajustar según el comportamiento observado en la salida de los MFCCs.
        
        if (val > 1.0f) val = 1.0f;     // Limitar a 1.0 para evitar valores extremos que puedan afectar la inferencia
        if (val < -1.0f) val = -1.0f;   // Limitar a -1.0 para evitar valores extremos que puedan afectar la inferencia
        
        normalized_signal[j] = val;   // La normalización final da como resultado una señal que oscila aproximadamente entre -1.0 y 1.0, con un énfasis en las frecuencias relevantes para la detección de emociones en la voz.
    }
}//fin audio_normalized

// Función para interpretar la salida del modelo de 0.0 a 1.0 y determinar la emoción detectada, además de imprimir una barra visual y el porcentaje de confianza en la emoción "angry"
int interprets_output(TfLiteTensor *outputTensor) {
    if (!outputTensor || !outputTensor->data.f) return 0;
    
    // 1. Obtener el valor real crudo (0.0 a 1.0) en unidad de probabilidad para "angry"
    float val_angry = outputTensor->data.f[0];  
     
    
    // 2. Determinar el resultado de la emocion detectada
    int result = (val_angry >= 0.70)? 1 : 0;    //con el umbral estaba a 0.85 vamos a probar con uno nuevo

    // 3. Crear la barra visual [##########----------] que representa el porcentaje de "angry"
    char barra[23];   // [ + 20 + ] + \0
    barra[0] = '[';

    int longitudBarra = 20;
    int rellenos = (int)(val_angry * longitudBarra);

    for (int j = 0; j < longitudBarra; j++) {
        barra[j+1] = (j < rellenos) ? '#' : '-';
    }

    barra[21] = ']';
    barra[22] = '\0';

    // --- SALIDA AL MONITOR SERIAL ---
    // salida original : en unidad de probabilidad y emoción detectada
    Serial.printf("Salida: %.6f - %s\n", val_angry, emotion_vec[result].c_str()); //muestra el valor de angry y la emoción detectada (angry u other)
    
    //monitor de porcentajes y escala visual tipo barra para entender mejor la confianza del modelo en su predicción
    Serial.printf("\nANGRY: %d%%  %s\n", (int)(val_angry*100), barra);
    
    return result;
}//fin interprets_output



// FUNCIONES DE MEMORIA PSRAM
// Función para asignar una matriz de MFCCs en PSRAM
float** allocate_mfcc_matrix_psram(int rows, int cols) {
  float **matrix = new float*[rows];      // Asignar el arreglo de punteros a filas en RAM interna
  for (int j = 0; j < rows; j++) {       // Asignar cada fila de la matriz en PSRAM
    matrix[j] = (float*)heap_caps_malloc(cols * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);    // Asignar cada fila de la matriz en PSRAM
    if (!matrix[j]) return nullptr;       // Si falla la asignación, retornar nullptr para manejar el error en el caller
  }
  return matrix;    // Retornar el puntero a la matriz de MFCCs asignada en PSRAM
}

// Liberar la matriz de MFCCs en PSRAM
void free_mfcc_matrix_psram(float **matrix, int rows) {  // Verificar que el puntero no sea nulo antes de liberar
  if (!matrix) return;  // Verificar que el puntero no sea nulo antes de liberar
  for (int j = 0; j < rows; j++) if (matrix[j]) heap_caps_free(matrix[j]); // Liberar cada fila de la matriz
  delete[] matrix; // Liberar el arreglo de punteros a filas
}

// Funciones para arrays de floats e int32_t en PSRAM
float* allocate_float_array_psram(size_t size) {
  return (float*)heap_caps_malloc(size * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

// Liberar array de floats en PSRAM
void free_float_array_psram(float *arr) { if (arr) heap_caps_free(arr); }
// Función para asignar un array de int32_t en PSRAM
int32_t* allocate_int32_array_psram(size_t size) {
  return (int32_t*)heap_caps_malloc(size * sizeof(int32_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

// Liberar array de int32_t en PSRAM
void free_int32_array_psram(int32_t *arr) { if (arr) heap_caps_free(arr); }


// NOTAS

// bool success = nn->predict(); 
// Ejecuta la inferencia del modelo, es decir, corre la red neuronal usando los MFCC calculados.

// TfLiteTensor *outputTensor = nn->getOutputTensor();  
// Obtiene el tensor de salida del modelo donde se almacena la predicción.

// float val_angry = outputTensor->data.f[0];   
// Obtiene el valor de salida del modelo para "angry". 
// Este valor está en formato de probabilidad entre 0.0 y 1.0.

// en la impresion se multiplica  (int)(val_angry * 100);  
// Convierte la probabilidad de enojo (0.0–1.0) a un porcentaje (0–100).