Compact Near Real-Time Anger Detection System through Voice Analysis Implemented on ESP32 Microcontroller

This project presents the development of a compact device designed to detect the emotion of anger through real-time voice analysis.

Creating a compact solution poses significant challenges due to the computational and memory limitations of small embedded processors. While previous research has achieved high accuracy (up to 95%) in emotion detection using deep neural networks with millions of parameters, such architectures are impractical for deployment on low-power hardware platforms.

To address this limitation, a lightweight convolutional neural network inspired by the Light-CNN architecture was designed. The proposed model focuses on maintaining low computational complexity while preserving sufficient accuracy for real-time inference on embedded systems.

Using TensorFlow Lite, the trained neural network was converted and optimized for execution on a microcontroller. The final model contains 21,697 trainable parameters and requires approximately 10 million floating point operations (FLOPS) per inference, making it suitable for edge AI applications.

The neural network was trained using Mel-Frequency Cepstral Coefficients (MFCC) extracted from the Emotional Speech Database (ESD).

The implementation was carried out on an ESP32-WROVER-E microcontroller, featuring:

Audio sample acquisition using a microphone
Real-time MFCC feature extraction
Neural network inference using TensorFlow Lite
Visualization of results on an LED matrix indicating the presence or absence of anger

The system achieves a total processing time of approximately 409 ms per inference cycle, including MFCC computation and neural network inference, demonstrating its feasibility for near real-time emotion detection on resource-constrained hardware.

Audio Analysis

The audio analysis process extracts Mel-Frequency Cepstral Coefficients (MFCCs), which are widely used in speech processing because they approximate the human auditory perception of sound.

The main steps are described below.

1. Pre-emphasis

A pre-emphasis filter is applied to the audio signal to amplify high-frequency components that are naturally attenuated in speech signals, improving phoneme discrimination and feature extraction.

2. Framing and Windowing

The audio signal is divided into overlapping frames using a Hamming window. This reduces discontinuities at the edges of each frame and improves frequency analysis stability.

3. Spectrogram and Fast Fourier Transform (FFT)

Each frame is transformed into the frequency domain using the Fast Fourier Transform (FFT). This produces the signal spectrogram, representing the distribution of energy across frequencies.

4. Mel Scale and Triangular Filters

To better match human auditory perception, frequencies are mapped to the Mel scale, which emphasizes lower frequencies where most speech information resides.

A bank of triangular filters is applied to the spectrum to extract energy bands relevant to speech frequencies.

5. Logarithmic Transformation

The filter bank energies are converted into the logarithmic scale, approximating the human perception of sound loudness.

6. Discrete Cosine Transform (DCT)

Finally, a Discrete Cosine Transform (DCT) is applied to decorrelate the filter outputs and obtain the MFCC coefficients, which provide a compact representation of the audio signal.

MFCC Extraction Pipeline

The MFCC extraction process follows this sequence:

Apply pre-emphasis filter
Segment audio into frames using a Hamming window
Compute the FFT of each frame
Apply Mel-scale triangular filter banks
Apply logarithmic compression
Compute the Discrete Cosine Transform (DCT)

The resulting MFCC feature matrix is used as input for the neural network.

Model: MFCC-LightCNN

The AI model developed for this project, named MFCC-LightCNN, is a lightweight convolutional neural network inspired by the Light-CNN architecture, adapted for speech emotion recognition and optimized for execution on embedded hardware.

Instead of processing images as in the original Light-CNN design, the model processes MFCC spectrogram representations of audio signals, allowing convolutional layers to capture temporal and spectral patterns associated with emotional speech.

Key Features
Lightweight Architecture

The proposed model contains 21,697 trainable parameters, enabling efficient deployment on the ESP32-WROVER-E microcontroller.

For comparison:

Architecture	Parameters
VGG16	138 million
LightCNN9	~5 million
MFCC-LightCNN (proposed)	21,697

This reduction allows the model to run within the memory constraints of embedded systems.

Optimized Input Representation

Instead of the standard VGG16 input (224×224×3), the model uses MFCC features with the following input size:

20 × 124 × 1

This representation captures both spectral and temporal characteristics of speech while keeping the input size small.

Architecture Overview

The MFCC-LightCNN architecture consists of the following layers:

Feature Extraction Stage

Convolutional layer
32 filters (3×3)
ReLU activation
Output: 20×124×32
Max pooling layer
Two convolutional layers
16 filters (3×3)
ReLU activation
Max pooling layer

This stage extracts hierarchical spectral features from the MFCC input.

Embedding Layer

The convolutional feature maps are flattened and passed through a dense layer:

Dense layer with 32 neurons, producing a compact feature embedding.

Classification Head

The classification stage contains:

Dense layer (32 neurons)
Dropout regularization
Final sigmoid layer producing a probability between 0 and 1.

The output represents the likelihood of the presence of anger in the analyzed audio segment.

Training Hyperparameters

The model was trained using the following configuration:

Learning rate

0.00047

Optimizer

Adam

Epochs

250

Batch size

490

Loss function

Binary Crossentropy

Metric

Accuracy

Training was performed using TensorFlow/Keras, and the final model was converted to TensorFlow Lite for embedded deployment.

Model Performance

The embedded system processes audio segments and performs inference with the following timing characteristics:

Process	Time
MFCC computation	~190 ms
Neural network inference	~219 ms
Total processing time	~409 ms

The system operates within the memory constraints of the ESP32:

Free RAM: ~229 KB
Free PSRAM: ~7 MB

These results demonstrate that the proposed MFCC-LightCNN architecture enables near real-time anger detection on a microcontroller platform, making it suitable for edge AI applications involving speech emotion recognition.