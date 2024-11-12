# DeepFusionNet-17: Attention-Enhanced Convolutional Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Models Overview](#models-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Documentation and Comments](#documentation-and-comments)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

**DeepFusionNet-17** is a comprehensive deep learning framework that integrates various attention mechanisms into convolutional and recurrent neural networks. The repository explores and enhances feature extraction capabilities across different data modalities and dimensions (1D, 2D, 3D) by combining:

- Convolutional Neural Networks (CNN)
- Bidirectional Long Short-Term Memory Networks (Bi-LSTM)
- Attention Mechanisms (Self-Attention, Bahdanau, Multi-Head Attention)
- Residual Blocks
- Mix of Experts (MOEs)

---

## Features

- **Multiple Model Architectures**: 17 different models combining various deep learning components.
- **Attention Mechanisms**: Implementation of several attention types to enhance model performance.
- **Residual Connections**: Integrated residual blocks for deeper network training.
- **Support for Different Data Dimensions**: Models designed for 1D, 2D, and 3D data.
- **Extensive Documentation**: Comprehensive comments and documentation in code, especially in version 17.
- **Easy to Extend**: Modular design allows for easy customization and extension.

---

## Models Overview

This repository includes 17 model architectures with various combinations of:

- **Convolutional Layers**: CNNs in 1D, 2D, and 3D configurations.
- **Bi-LSTM Layers**: Recurrent layers to capture sequential dependencies.
- **Attention Mechanisms**: Includes simple self-attention, additive attention (Bahdanau), multi-head attention, and mix of experts (MOEs).
- **Residual Connections**: Integrated in some versions for enhanced feature retention.

---

## Installation

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.x and/or PyTorch 1.x
- Git

### Clone the Repository

```bash
git clone https://github.com/alirezaabbaszadeh/ComprehensiveDeepLearningFramework-17.git
cd ComprehensiveDeepLearningFramework-17
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training a Model

To train a specific model version:

```python
from models import ModelBuilder

# Initialize the model builder with the desired version
model = ModelBuilder(version=17)  # Replace with the desired version number

# Train the model
model.train()
```

### Evaluating a Model

```python
# Evaluate the model
model.evaluate()
```

### Predicting with a Model

```python
# Make predictions
predictions = model.predict(test_data)
```

---

## Model Architectures

Below are the architectures and main components in each version:

### 1D Models

1. **Model 1**
   - **CNN-1D**
   - **Bi-LSTM**
   - **Simple Self-Attention**

2. **Model 11**
   - **Attention-Augmented Convolutional Networks 1D (AACN1D)**
   - **Bi-LSTM**
   - **Simple Self-Attention**

3. **Model 12**
   - **Residual Block - AACN1D (RB-AACN-1D)**
   - **Bi-LSTM**
   - **Simple Self-Attention**

4. **Model 13**
   - **RB-AACN-1D**
   - **Bi-LSTM**
   - **Additive Attention (Bahdanau)**

5. **Model 14**
   - **RB-AACN-1D**
   - **Bi-LSTM**
   - **Multi-Head Attention (MHA)**

6. **Model 15**
   - **RB-AACN-1D**
   - **Bi-LSTM**
   - **MHA**
   - **Fully Connected Layer (FCL)**

7. **Model 16**
   - **RB-AACN-1D**
   - **Bi-LSTM**
   - **MHA**
   - **Mix of Experts (MOEs)**

8. **Model 17**
   - **RB-AACN-1D**
   - **Bi-LSTM**
   - **MHA**
   - **MOEs**
   - **FCL**

### 2D and 3D Models

1. **Model 2**
   - **CNN-2D**
   - **Bi-LSTM**
   - **Simple Self-Attention**

2. **Model 3**
   - **CNN-3D**
   - **Bi-LSTM**
   - **Simple Self-Attention**

3. **Model 4**
   - **Attention-Augmented Convolutional Networks 3D (AACN3D)**
   - **Bi-LSTM**
   - **Simple Self-Attention**

4. **Model 5**
   - **Residual Block - AACN3D (RB-AACN-3D)**
   - **Bi-LSTM**
   - **Simple Self-Attention**

5. **Model 6**
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Additive Attention (Bahdanau)**

6. **Model 7**
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **Multi-Head Attention (MHA)**

7. **Model 8**
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **MHA**
   - **Fully Connected Layer (FCL)**

8. **Model 9**
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **MHA**
   - **Mix of Experts (MOEs)**

9. **Model 10**
   - **RB-AACN-3D**
   - **Bi-LSTM**
   - **MHA**
   - **MOEs**
   - **FCL**

### Key Classes

- `ModelBuilder`: The main class that assembles each model version. Version 17 includes detailed documentation and comments to guide understanding and modifications.

---

## Documentation and Comments

- **Version 17**: Contains complete documentation and comments in the `ModelBuilder` class. Each function includes descriptions of inputs, outputs, and logic, focusing on clarity for those extending or modifying model behavior.
- **Docstrings**: All classes and methods include docstrings for better code readability and maintainability.
- **Inline Comments**: Important code sections have inline comments explaining the logic and implementation details.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click on the 'Fork' button at the top right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/ComprehensiveDeepLearningFramework-17.git
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Add new features or fix bugs.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add your commit message here"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   Go to the original repository and click on 'New Pull Request'.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to all contributors and to the research community in the fields of attention mechanisms and advanced CNN architectures.

- **Attention Mechanisms**: Vaswani et al. (2017), "Attention Is All You Need"
- **Residual Networks**: He et al. (2015), "Deep Residual Learning for Image Recognition"
- **Mix of Experts**: Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

---

Feel free to explore, use, and contribute to this repository to advance the development of attention-enhanced convolutional models!

# Contact

For any questions or inquiries, please contact:

- **Alireza Abbaszadeh**
- Email: [link.aabz@gmail.com](mailto:link.aabz@gmail.com)
- GitHub: [alirezaabbaszadeh](https://github.com/alirezaabbaszadeh)

---

*This README was generated to provide a comprehensive overview of the DeepFusionNet-17 project. It includes all necessary information to get started and contribute to the project.*

# Note

Please replace placeholder information like `<repository_url>`, `<repository_folder>`, and `[your.email@example.com]` with your actual repository URL, local folder name, and contact email.
