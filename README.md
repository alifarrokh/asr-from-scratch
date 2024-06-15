
# Speech Recognition Models from Scratch

This repository contains implementations of two popular end-to-end speech recognition models:

- **LAS (Listen, Attend, Spell)**
- **Deep Speech 2**

Both models have been implemented from scratch in **PyTorch** and **PyTorch Lightning**, focusing on replicating their original architectures and functionalities.

## Project Overview

### Deep Speech 2
Deep Speech 2, introduced by Dario Amodei et al. in their paper ["Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"](https://arxiv.org/abs/1512.02595), is a convolutional recurrent neural network (CRNN) that efficiently maps speech signals to text transcriptions. The model is trained using connectionist temporal classification (CTC) loss, allowing it to learn directly from the sequential audio input without the need for explicit alignment.

### LAS (Listen, Attend, Spell)
LAS, proposed by William Chan et al. in their paper ["Listen, Attend and Spell"](https://arxiv.org/abs/1508.01211), is an encoder-decoder model with an attention mechanism for end-to-end automatic speech recognition (ASR). The encoder processes the input audio features, while the decoder generates the transcriptions character by character, utilizing the attention module to align input features with output tokens.

## Current Status

- **Implementation**: Both LAS and Deep Speech 2 models are implemented from scratch in PyTorch and PyTorch Lightning in this repository. 
- **Training**: Due to resource constraints, only simple experiments have been conducted to validate the basic functionality of the implementations. The models have not yet been trained on publicly available datasets. 

## Planned Improvements

- **Training**: The plan is to train these models on a subset of the **LibriSpeech** dataset and publish the weights on Hugging Face Hub, making them accessible for further experimentation and use.
- **Directory Restructuring**: Improve the directory structure for better modularity and maintainability.
