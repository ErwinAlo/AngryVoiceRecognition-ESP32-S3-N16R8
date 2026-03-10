#include <Arduino.h>
#include <esp_heap_caps.h>
#include "NeuralNetwork.h"
#include "model_data.h"

// Tamaño óptimo del arena para tu modelo (~400KB)
// TensorFlow Lite Micro necesita ~100-200KB adicionales para activaciones
// Total recomendado: ~600KB (mucho menos que los 2MB que tenías)
const size_t kArenaSize = 600000;  // 600 KB es suficiente  antes 2000000

NeuralNetwork::NeuralNetwork() : 
    model(nullptr), 
    interpreter(nullptr), 
    input(nullptr), 
    output(nullptr),
    tensor_arena(nullptr),
    arena_size(kArenaSize),
    aligned_arena(nullptr)
{
}

NeuralNetwork::~NeuralNetwork()
{
    // Liberar memoria PSRAM al destruir el objeto
    if (tensor_arena) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
        Serial.println("PSRAM liberada");
    }
}

bool NeuralNetwork::begin()
{
    // 1. Verificar que PSRAM está disponible
    if (!psramFound()) {
        Serial.println("ERROR: PSRAM no detectada!");
        Serial.println("Asegúrate de que tu placa tiene PSRAM y está habilitada en platformio.ini");
        return false;
    }
    
    size_t psramSize = ESP.getPsramSize();
    size_t freePsram = ESP.getFreePsram();
    Serial.printf("PSRAM detectada: Total=%d KB, Libre=%d KB\n", 
                  psramSize/1024, freePsram/1024);

    // 2. Cargar el modelo
    model = tflite::GetModel(modelo_tesis_binario_Angry_finalv5_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("ERROR: Versión del esquema no compatible. "
                      "Modelo=%d, Esperado=%d\n", 
                      model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    Serial.println("Modelo cargado correctamente");

    // 3. Registrar operaciones necesarias
    resolver_instance.AddFullyConnected();
    resolver_instance.AddLogistic();
    resolver_instance.AddReshape();
    resolver_instance.AddConv2D();
    resolver_instance.AddMaxPool2D();
    resolver_instance.AddL2Normalization();
    Serial.println("Operaciones registradas");

    // 4. Asignar memoria en PSRAM con espacio para alineación
    // Reservamos kArenaSize + 15 bytes para poder alinear a 16 bytes
    size_t alloc_size = kArenaSize + 15;   
    
    tensor_arena = (uint8_t *)heap_caps_malloc(alloc_size,   //cambiado a alloc_size para incluir espacio de alineación
                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (!tensor_arena) {
        Serial.printf("ERROR: No se pudo asignar %d bytes en PSRAM\n", alloc_size);
        return false;
    }
    
    // Alinear a 16 bytes (requerido por TensorFlow Lite)
    aligned_arena = (uint8_t *)(((uintptr_t)tensor_arena + 15) & ~15);
    
    size_t alignment_offset = aligned_arena - tensor_arena;
    Serial.printf("Memoria PSRAM asignada: %d bytes (alineación: +%d bytes)\n", 
                  alloc_size, alignment_offset);

    // 5. Crear el intérprete (SIN static, como miembro de la clase)
    // Usamos placement new para crear el intérprete en memoria persistente
    static uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)];
    interpreter = new (interpreter_buffer) tflite::MicroInterpreter(
        model, resolver_instance, aligned_arena, kArenaSize, &error_reporter_instance);

    // 6. Reservar memoria para tensores
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() falló");
        return false;
    }
    Serial.println("Tensores asignados correctamente");

    // 7. Obtener punteros a tensores de entrada/salida
    input = interpreter->input(0);
    output = interpreter->output(0);

    if (!input || !output) {
        Serial.println("ERROR: No se pudieron obtener tensores de entrada/salida");
        return false;
    }

    // 8. Configurar dimensiones del tensor de entrada
    // Tu modelo espera: [1, 20, 63, 1] = [batch, mel_bands, time_windows, channels]
    input->dims->data[0] = 1;           // batch size
    input->dims->data[1] = 20;          // MEL_BANDS (frecuencias)
    input->dims->data[2] = 63;          // NUMBER_OF_WINDOWS (tiempo)
    input->dims->data[3] = 1;           // canales (escala de grises)
    input->type = kTfLiteFloat32;

    Serial.println("Red neuronal inicializada correctamente en PSRAM");
    printMemoryInfo();
    
    return true;
}

float *NeuralNetwork::getInputBuffer()
{
    if (!input) return nullptr;
    return input->data.f;
}

TfLiteTensor* NeuralNetwork::getOutputTensor() 
{
    return output;
}

bool NeuralNetwork::predict()
{
    if (!interpreter) {
        Serial.println("ERROR: Interpreter no inicializado");
        return false;
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERROR: Invoke() falló");
        return false;
    }
    return true;
}

//  imprimir info de memoria (RAM interna y PSRAM)
void NeuralNetwork::printMemoryInfo()   // los :: indican que es un método de la clase NeuralNetwork
{
    Serial.println("\n--- Información de Memoria ---");
    Serial.printf("RAM Interna Total: %d KB\n", ESP.getHeapSize() / 1024);
    Serial.printf("RAM Interna Libre: %d KB\n", ESP.getFreeHeap() / 1024);
    Serial.printf("PSRAM Total: %d KB\n", ESP.getPsramSize() / 1024);
    Serial.printf("PSRAM Libre: %d KB\n", ESP.getFreePsram() / 1024);
    Serial.printf("Tamaño Arena TFLite: %d KB (en PSRAM)\n", kArenaSize / 1024);
    Serial.println("------------------------------");
}