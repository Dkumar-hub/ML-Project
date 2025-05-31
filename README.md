# ML-Project
Human Sign Language Recognition System Using LSTM.
This project implements a deep learning model for recognizing sign language gestures based on hand landmark sequences extracted using MediaPipe.

Model Overview
The model uses a multi-layer LSTM (Long Short-Term Memory) architecture enhanced with an attention mechanism to capture temporal dependencies and highlight important frames in a gesture sequence.

Input: Sequences of 30 frames, each frame containing 126 keypoints (3D coordinates of 21 hand landmarks per hand).

Layers:

Two stacked LSTM layers with 256 and 128 units, followed by batch normalization and dropout for regularization.

An attention mechanism generating weights per frame to focus on key moments.

A third LSTM layer processes the attention-weighted sequence to extract meaningful features.

Fully connected dense layers for final classification.

Output: Softmax layer predicting one of 5 ASL sign language classes: Hello, Goodbye, Sorry, Thank you, No.

Features
Captures spatial-temporal features from hand movement sequences.

Custom attention layer improves focus on relevant gesture frames.

Robust to noisy or incomplete hand landmark detections.

Suitable for real-time sign language recognition applications.

Usage
The model takes processed hand landmark sequences and predicts the corresponding sign gesture in real-time.

