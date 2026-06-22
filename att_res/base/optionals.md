# Optional Configurations in the AERC Model Architecture

This directory contains two implementations of the Attention-Enhanced Reservoir Computing (AERC) model:
1. **[aerc_simplified.py](file:///home/medlar/Projects/screening/att_res/base/aerc_simplified.py)**: The base model without Intrinsic Plasticity pre-training.
2. **[aerc_ip.py](file:///home/medlar/Projects/screening/att_res/base/aerc_ip.py)**: The model with Intrinsic Plasticity (IP) pre-training capability.

---

## 1. Removed Optional Components (in both implementations)

To simplify the model architecture, the following features from the original codebase have been completely removed or hardcoded to their default/inactive state in both versions:

### Leaky Integration / Leaky Neurons (`leaking_rate`)
- **Original Behavior**: Leaky integrations scaled the activations of recurrent reservoir states using a leaking rate parameter $\alpha \in (0, 1]$.
- **New Behavior**: Removed. The reservoir dynamics run without leakiness (effectively $\alpha = 1.0$).

### Output Feedback Connections (`fb_scaling` / `W_fb`)
- **Original Behavior**: Fed the previous reservoir states back into the input state computation via a feedback matrix.
- **New Behavior**: Removed. There is no feedback loop (effectively feedback scaling is 0.0).

### Gate Activation Choice (`activation`)
- **Original Behavior**: The gate network (connecting the reservoir states to the attention weighting mechanism) could select between `silu`, `tanh`, or `relu`.
- **New Behavior**: Removed. Hardcoded to use `silu` (SiLU/Swish) exclusively.

### Two-Phase Training / Base Reservoir Fitting (Ridge Regression)
- **Original Behavior**: Model training occurred in two phases. Phase 1 used analytical **Ridge Regression** to solve and freeze `static_head`. Phase 2 trained the attention-based correction layer using backpropagation.
- **New Behavior**: Removed. Both models are trained fully end-to-end (all trainable parameters are optimized together using backpropagation via gradient descent/AdamW).

### Attention Dropout (`dropout`)
- **Original Behavior**: Applied optional dropout regularization on gate network activations.
- **New Behavior**: Removed. The dropout layer is deleted and activations are passed directly.

---

## 2. Intrinsic Plasticity pre-training (`pretrain_reservoir_ip`)

- **In `aerc_simplified.py`**: Completely removed. The reservoir is initialized using purely random recurrent connection weights (scaled for the Echo State Property) and remains static throughout.
- **In `aerc_ip.py`**: Retained. Before the main training phase, the reservoir is pre-trained using a data-driven Intrinsic Plasticity (IP) rule to adapt neuron statistics to match target Gaussian statistics. Once pre-trained, the scaling and biases are folded into the fixed reservoir parameters, and the model is trained end-to-end.

---

## 3. Retained Configurable Hyperparameters (in both implementations)

These parameters remain configurable as constructor arguments in both `AERC` classes:

### Spectral Radius (`spectral_radius`)
- **Description**: Scales the largest absolute eigenvalue of the recurrent weight matrix `W_hh` to control memory and echo state properties.
- **Default**: `0.95`.

### Input Embedding Dimension (`d_e`)
- **Description**: Dimension of the fixed character embeddings projecting input tokens.
- **Default**: `16`.

### Reservoir Size (`N`)
- **Description**: Number of recurrent units (neurons) in the fixed reservoir.
- **Default**: `147` (selected to target ~155k parameters).

### Attention Subspace Dimension (`H`)
- **Description**: The dimension of the low-rank attention subspace mapping.
- **Default**: `30` (selected to target ~155k parameters).

### Vocabulary Size (`vocab_size`)
- **Description**: The size of the character vocabulary.
- **Default**: Configured dynamically based on the dataset (e.g. `65` for standard case-sensitive `tinyshakespeare.txt`).
