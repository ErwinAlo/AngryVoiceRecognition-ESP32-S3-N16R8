#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include <stdint.h>
#include <Arduino.h>

// Evita conflictos entre las definiciones de Arduino y TensorFlow
#ifdef DEFAULT
#undef DEFAULT
#endif

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

struct TfLiteTensor;

class NeuralNetwork
{
private:
    tflite::MicroErrorReporter error_reporter_instance;
    tflite::MicroMutableOpResolver<10> resolver_instance;
    
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    
    // Buffer de memoria para tensores (en PSRAM)
    uint8_t *tensor_arena;
    size_t arena_size;
    
    // Buffer alineado (16 bytes) para TensorFlow Lite
    uint8_t *aligned_arena;

public:
    NeuralNetwork();
    ~NeuralNetwork();  // Destructor para liberar PSRAM
    
    bool begin();  // Inicialización separada del constructor
    bool isReady() const { return interpreter != nullptr; }
    
    float *getInputBuffer();
    TfLiteTensor *getOutputTensor();
    bool predict();
    
    // Información de memoria
    void printMemoryInfo();
};

#endif