# Neural Network from Scratch (NumPy)

## Overview
This project implements a fully connected deep neural network **from scratch**
using only NumPy, without relying on deep learning frameworks.

The goal is to understand and demonstrate the **mathematical intuition**
behind forward propagation, backpropagation, and gradient-based learning.

---

## Model Architecture
Input → 64 → 32 → 16 → Output (Binary)

- Input layer : medical features
- Hidden layers : ReLU activation
- Output layer : Sigmoid activation
- Loss : Binary Cross Entropy

![Neural Network Architecture](images/nn_architecture.png)

---

## Mathematical Intuition

### Forward Propagation
Each layer performs:
Z = XW + b  
A = activation(Z)

ReLU introduces non-linearity, allowing the network to model complex patterns.

### Loss Function
Binary Cross Entropy measures how well predicted probabilities
match actual labels.

### Backpropagation
Gradients are computed using the chain rule, flowing from the output
layer back to the input layer to update parameters.

---

## Training
- Optimizer: Gradient Descent
- Learning Rate: configurable
- Dataset: Breast Cancer Wisconsin (cleaned)

---

## Key Learnings
- Importance of weight initialization
- Handling class imbalance
- Debugging model behavior vs code correctness

---

## Future Work
- Mini-batch training
- Regularization
- Visualization
- Extension to multi-class classification
