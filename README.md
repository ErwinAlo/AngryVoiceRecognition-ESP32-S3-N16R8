Compact Near Real-Time Anger Detection System through Voice Analysis Implemented on ESP32-N16R8

This project presents the development of a compact embedded system designed to detect the emotion of anger through near real-time voice analysis.

Designing speech emotion recognition systems for embedded platforms presents significant challenges due to the limited computational resources and memory constraints of microcontrollers. Many modern approaches rely on deep neural networks containing millions of parameters, which require powerful GPUs and large memory resources. Such architectures are impractical for deployment on low-power edge devices.

To address these limitations, this work proposes a lightweight convolutional neural network architecture specifically designed for embedded inference. The model focuses on maintaining low computational complexity while preserving sufficient accuracy for speech emotion detection.

The trained neural network was converted to TensorFlow Lite for Microcontrollers, allowing it to run directly on an embedded platform. The final model contains approximately 21,697 trainable parameters and requires roughly 10 million floating-point operations (FLOPs) per inference.

Speech signals are processed using Mel-Frequency Cepstral Coefficients (MFCC) as input features.

The system was implemented on an ESP32-N16R8 microcontroller, which provides:

16 MB Flash memory
8 MB PSRAM
sufficient resources for lightweight embedded machine learning

The embedded system performs the following tasks:

• Audio acquisition using a microphone
• Real-time MFCC feature extraction
• Neural network inference using TensorFlow Lite
• Visualization of results on an LED matrix indicating the detected emotional state

Experimental measurements show that the complete processing pipeline requires approximately 409 ms per inference cycle, including MFCC extraction and neural network evaluation. These results demonstrate the feasibility of implementing speech emotion recognition in near real-time on resource-constrained embedded hardware.

Audio Analysis

The audio processing stage extracts Mel-Frequency Cepstral Coefficients (MFCCs) from speech signals. MFCC features are widely used in speech processing because they approximate the nonlinear frequency perception of the human auditory system.

These coefficients provide a compact representation of the spectral envelope of speech signals and are commonly used in applications such as speech recognition, speaker identification, and emotion recognition.

The MFCC extraction process involves several signal processing stages.

Pre-emphasis

A pre-emphasis filter is first applied to the speech signal. This filter amplifies high-frequency components that are naturally attenuated during speech production.

Enhancing these components improves the representation of phonetic information and increases the discriminative capability of the extracted features.

Framing and Windowing

The continuous audio signal is divided into short overlapping frames using a Hamming window. Windowing reduces spectral leakage and minimizes discontinuities at frame boundaries.

Within each frame, the speech signal can be considered approximately stationary, which allows reliable spectral analysis.

Fast Fourier Transform

Each windowed frame is transformed into the frequency domain using the Fast Fourier Transform (FFT).

The FFT converts the time-domain signal into its frequency representation, allowing the analysis of how signal energy is distributed across frequency components.

Mel Filter Bank

The frequency spectrum is then mapped to the Mel scale, which approximates the nonlinear frequency perception of human hearing.

A set of triangular filters is applied to the spectrum to compute the energy contained in perceptually relevant frequency bands.

Logarithmic Compression

The energies obtained from the Mel filter bank are converted to the logarithmic scale. This transformation approximates the nonlinear perception of sound loudness by the human auditory system.

Discrete Cosine Transform

Finally, a Discrete Cosine Transform (DCT) is applied to decorrelate the filter bank outputs and produce the MFCC coefficients.

This step generates a compact feature representation of the speech signal that is suitable for machine learning models.

MFCC Feature Extraction Configuration

The speech signals used in this project are sampled at 16 kHz, which captures the relevant frequency range of human speech while maintaining a manageable computational cost.

The audio is segmented into 2-second windows, producing speech segments containing 32,000 samples.

MFCC features are computed using the following parameters:

Sampling rate:
16,000 Hz

FFT size:
n_fft = 512

Hop length:
hop_length = 256

Number of Mel filters:
20

With this configuration, the MFCC extraction stage produces a feature matrix of size:

20 × 124

where:

20 corresponds to the number of MFCC coefficients
124 corresponds to the number of temporal frames extracted from the 2-second audio segment
Central Temporal Cropping

Although the MFCC extraction process initially produces 124 temporal frames, not all frames contribute equally to emotion recognition.

The beginning and ending regions of speech segments often contain silence, articulation transitions, or unstable phonetic patterns that may introduce noise in the feature representation.

To improve efficiency and reduce computational cost, a central temporal cropping strategy is applied.

This process selects the central 63 frames of the MFCC representation while preserving all spectral coefficients.

The MFCC feature matrix is therefore reduced from:

20 × 124 × 1

to

20 × 63 × 1

This reduction significantly decreases the number of convolution operations required during neural network inference.

Additionally, the central portion of speech segments usually contains the most stable phonetic content and emotional cues, improving the relevance of the extracted features.

The reduced input size also improves the feasibility of deploying the model on resource-constrained hardware platforms such as the ESP32-N16R8 microcontroller.

Justification of 2-Second Audio Segments

The system processes speech using 2-second audio segments.

This duration represents a compromise between capturing sufficient emotional context and maintaining real-time processing capability.

Emotional expression in speech is often conveyed through prosodic features, such as pitch variation, energy patterns, and temporal dynamics. These features occur over short time intervals rather than individual phonemes.

A 2-second segment provides enough temporal context to capture these patterns while keeping the computational requirements manageable for embedded deployment.

Since the full processing pipeline requires approximately 409 ms, each segment can be analyzed significantly faster than its duration. This enables continuous speech monitoring in near real-time conditions.

Model: MFCC-CNNv2 (Hybrid LightCNN-Inspired Architecture)

The artificial intelligence model developed in this work, referred to as MFCC-CNNv2, is a lightweight convolutional neural network designed for speech emotion recognition on embedded hardware.

The architecture is inspired by the LightCNN-9 model, originally developed for face recognition tasks. LightCNN architectures emphasize efficient convolutional structures that reduce the number of parameters while maintaining strong feature extraction capability.

In this work, the design philosophy of LightCNN-9 was adapted to create a hybrid convolutional architecture optimized for MFCC-based speech analysis.

Instead of processing image data, the network processes MFCC spectrogram representations of speech, allowing convolutional filters to capture both spectral and temporal patterns associated with emotional speech.

Lightweight Architecture

The proposed network contains approximately 21,697 trainable parameters, enabling efficient deployment on microcontroller platforms.

For comparison:

Architecture	Parameters
VGG16	138 million
LightCNN-9	~5 million
MFCC-CNNv2 (proposed)	21,697

This large reduction in parameter count allows the network to operate within the memory constraints of embedded systems.

Optimized Input Representation

The input to the neural network consists of MFCC features with dimensions:

20 × 63 × 1

This representation captures both:

spectral information of speech signals
temporal dynamics of emotional expression

while maintaining a compact input size suitable for embedded inference.

Architecture Overview

The MFCC-CNNv2 architecture consists of three main stages:

Feature extraction
Feature embedding
Classification
Feature Extraction Stage

The first stage extracts hierarchical features from the MFCC input using convolutional layers.

Layer sequence:

Convolutional layer
32 filters (3×3 kernel)
ReLU activation

Max pooling layer

Convolutional layer
16 filters (3×3 kernel)
ReLU activation

Convolutional layer
16 filters (3×3 kernel)
ReLU activation

Max pooling layer

These convolutional layers learn spectral-temporal patterns associated with emotional speech characteristics.

Embedding Layer

The extracted feature maps are flattened and passed through a dense layer:

Dense layer with 32 neurons

This layer produces a compact embedding vector representing the emotional characteristics of the speech segment.

Classification Head

The classification stage contains:

Dense layer (32 neurons)
Dropout regularization layer
Final sigmoid output neuron

The output neuron produces a probability between 0 and 1, representing the likelihood that the analyzed speech segment contains anger.

Training Configuration

The neural network was trained using the following configuration:

Learning rate:
0.00047

Optimizer:
Adam

Epochs:
250

Batch size:
490

Loss function:
Binary Crossentropy

Evaluation metric:
Accuracy

Training was performed using TensorFlow/Keras, and the final model was converted to TensorFlow Lite for embedded deployment.

Embedded System Performance

The embedded system processes audio segments and performs inference with the following timing characteristics.

Process	Time
MFCC computation	~190 ms
Neural network inference	~219 ms
Total processing time	~409 ms

Memory availability on the ESP32-N16R8 during execution:

Free RAM: ~229 KB
Free PSRAM: ~7 MB

These results demonstrate that the proposed MFCC-CNNv2 hybrid architecture enables near real-time anger detection on a microcontroller platform.

The system successfully integrates digital signal processing, machine learning, and embedded deployment, demonstrating the feasibility of implementing speech emotion recognition systems directly on low-power edge devices.