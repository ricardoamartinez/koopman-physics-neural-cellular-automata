# Koopman Physics Neural Cellular Automata

A real-time PyTorch implementation integrating Koopman-based Neural Cellular Automata with advanced components for physics simulation and learning.

## Overview

This project demonstrates a unified architecture combining multiple advanced machine learning components:

### Core Components

1. **Koopman-based Neural Cellular Automaton (NCA)**
   - Complex-valued latent states with unitary operator evolution
   - Skew-Hermitian generator with Cayley transform for stability
   - Norm-preserving dynamics for enhanced stability
   - 2D grid processing with neighbor interactions

2. **Multi-Wavelet Embedding/Decoding**
   - Multi-scale latent compression using Haar and Daubechies wavelets
   - Invertible representation with coarse approximation + detail subbands
   - Supports Haar (db1) and Daubechies-2 wavelets

3. **Complex-valued Hopfield Associative Memory**
   - Fast Hebbian outer-product updates with normalized patterns
   - Complex phase coding for enhanced information encoding
   - Pattern retrieval using complex inner products

4. **JEPA-style Contrastive Latent Prediction**
   - Joint Embedding Predictive Architecture approach
   - Mean-squared error (L2) and InfoNCE contrastive loss
   - Masked, delayed supervision for partial observation simulation

5. **Active Inference Loop with Online Constraint Enforcement**
   - Real-time learning without offline training epochs
   - Karush-Kuhn-Tucker style constraint enforcement
   - Lagrange multipliers for energy conservation

6. **Real-Time PyBullet Physics Integration**
   - Interactive physics simulations (bouncing balls, robotic arm)
   - Switchable environment scenarios
   - Real-time observation processing

7. **Interactive UI Controls**
   - PyBullet debug interface with sliders and buttons
   - Pause/resume, speed adjustment, component toggles
   - Model weight perturbation and save/load functionality

8. **Differentiable DSL Module**
   - Symbolic Domain-Specific Language for latent manipulation
   - Turing-complete with conditional branching and looping
   - Wavelet-based attention and constraint enforcement

## Features

- **Real-time Operation**: All components work together in an online learning loop
- **Interactive Control**: Full UI control over simulation parameters and model components
- **Modular Architecture**: Components can be toggled on/off during runtime
- **Physics Integration**: Direct integration with PyBullet for realistic physics
- **Advanced Learning**: Combines multiple state-of-the-art ML techniques
- **Visualization**: Real-time OpenCV visualization with status overlays

## Requirements

```
torch
numpy
pybullet
opencv-python
```

## Usage

```bash
python physics_nca.py
```

### Controls

- **Environment**: Switch between bouncing balls (0) and robotic arm (1)
- **Pause Simulation**: Pause/resume the physics and learning
- **Sim Speed**: Adjust simulation speed (0.1x to 2.0x)
- **Component Toggles**: Enable/disable NCA, Memory, DSL, and Learning
- **Model Controls**: Perturb weights, save/load model state

### Environments

1. **Bouncing Balls**: Physics simulation with elastic collisions in a bounded space
2. **Robotic Arm**: Franka Panda 7-DOF arm with oscillating joint motion

## Architecture Details

The system operates on a 480×480 RGB observation space, transformed via wavelets to a 240×240×12 latent space. The NCA processes this latent grid with complex-valued states, while the Hopfield memory stores coarse patterns for contrastive learning.

The active inference loop continuously:
1. Observes the current environment state
2. Predicts the next latent state using the NCA
3. Computes prediction errors and updates model parameters
4. Applies constraint enforcement via Lagrange multipliers
5. Visualizes results and accepts user input

## Key Innovations

- **Unitary Evolution**: Ensures stable, energy-preserving dynamics
- **Multi-scale Processing**: Wavelet embedding captures both coarse and fine features
- **Complex Memory**: Phase-coded associative memory for enhanced pattern storage
- **Real-time Learning**: Online parameter updates without batch training
- **Interactive Research**: Full runtime control for experimentation

## License

This project is open source and available under the MIT License. 