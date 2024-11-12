# DeepFusionNet-17
 Residual Block  MultHA Convolutional-BiLSTM-Attention Framework

 
# ModelBuilder README

This repository contains multiple deep learning models designed for sequence and spatial-temporal data processing. The models combine Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory networks (Bi-LSTM), and various attention mechanisms to enhance model performance on complex tasks.

## Model Architectures

Each model configuration utilizes different network components and attention layers. Below is a summary of the models included in the repository:

### 1. **1D Convolutional and Attention Models**

1. **Model 1**  
   - **CNN-1D**
   - **Bi-LSTM**
   - **Simple Attention (Self-Attention)**

2. **Model 2**  
   - **CNN-2D**
   - **Bi-LSTM**
   - **Simple Attention (Self-Attention)**

3. **Model 3**  
   - **CNN-3D**
   - **Bi-LSTM**
   - **Simple Attention (Self-Attention)**

4. **Model 4**  
   - **Attention-Augmented Convolutional Networks 3D (AACN3D)**
   - **Bi-LSTM**
   - **Simple Attention (Self-Attention)**

5. **Model 5**  
   - **Residual Block - AACN3D (RB-AACN-3D)**
   - **Bi-LSTM**
   - **Simple Attention (Self-Attention)**

6. **Model 6**  
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Additive Attention (Bahdanau)**

7. **Model 7**  
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Multi-Head Attention (MHA)**

8. **Model 8**  
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Multi-Head Attention (MHA)**
   - **Fully Connected Layer (FCL)**

9. **Model 9**  
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Multi-Head Attention (MHA)**
   - **Mix of Experts (MOE)**

10. **Model 10**  
    - **RB-AACN-3D**
    - **Bi-LSTM**
    - **Multi-Head Attention (MHA)**
    - **MOE**
    - **Fully Connected Layer (FCL)**

### 2. **1D Attention-Augmented Convolutional Networks**

11. **Model 11**  
    - **Attention-Augmented Convolutional Networks 1D (AACN1D)**
    - **Bi-LSTM**
    - **Simple Attention (Self-Attention)**

12. **Model 12**  
    - **Residual Block - AACN1D (RB-AACN-1D)**
    - **Bi-LSTM**
    - **Simple Attention (Self-Attention)**

13. **Model 13**  
    - **RB-AACN-1D**
    - **Bi-LSTM**
    - **Additive Attention (Bahdanau)**

14. **Model 14**  
    - **RB-AACN-1D**
    - **Bi-LSTM**
    - **Multi-Head Attention (MHA)**

15. **Model 15**  
    - **RB-AACN-1D**
    - **Bi-LSTM**
    - **Multi-Head Attention (MHA)**
    - **Fully Connected Layer (FCL)**

16. **Model 16**  
    - **RB-AACN-1D**
    - **Bi-LSTM**
    - **Multi-Head Attention (MHA)**
    - **Mix of Experts (MOE)**

17. **Model 17**  
    - **RB-AACN-1D**
    - **Bi-LSTM**
    - **Multi-Head Attention (MHA)**
    - **MOE**
    - **Fully Connected Layer (FCL)**

### Detailed Documentation

Model **version 17** in the `ModelBuilder` class includes comprehensive documentation and in-line comments to help users understand the implementation and usage of various layers and architectures.

## Installation

To use these models, clone the repository and install the required dependencies.

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
