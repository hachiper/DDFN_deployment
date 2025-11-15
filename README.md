# USFFNet Embedded Deployment Guide

This document describes how to deploy the **Uncertainty-Aware Spatial–Frequency Fusion Network (USFFNet)** on embedded hardware for real-time **State-of-Charge (SoC)** estimation in battery management systems (BMS).

USFFNet is designed to be compact and computationally efficient, making it suitable for **automotive-grade microcontrollers (MCUs)** and **low-power embedded SoCs**.

---

## 1. Model Overview

USFFNet is a dual-domain neural network that fuses temporal and frequency features and outputs **SoC and evidential uncertainty** (Normal–Inverse-Gamma parameters).

Typical characteristics of the trained model (reference implementation):

- **Task**: Real-time SoC estimation with uncertainty for BMS  
- **Architecture** (high level):
  - Temporal branch: 1D CNN-based encoder on time-domain sequences  
  - Frequency branch: 1D CNN-based encoder on FFT-based spectral features  
  - Spatial–Frequency Fusion blocks  
  - Evidential regression head (NIG) → outputs γ, ν, α, β  
- **Parameter count**: ≈ 2.38 × 10⁴ trainable parameters  
- **Checkpoint size**: ≈ 93 KB (FP32 weights)  
- **Runtime memory**: < 0.3 MB for batch size 1 (single input window)  
- **Computational complexity**: On the order of 10⁵ FLOPs per forward pass  

These figures are **approximate** and depend on the exact configuration (e.g., number of channels, kernel sizes). They show that USFFNet is well suited for **1–10 Hz SoC update rates**, which are typical in BMS applications.

---

## 2. Data Interface and Preprocessing

### 2.1 Input Signals

USFFNet operates on sliding windows of sensor measurements. A typical input window may contain:

- Cell/pack voltage \( V(t) \)  
- Current \( I(t) \) (often normalized in C-rate)  
- (Optional) Temperature \( T(t) \)  
- Time index or implicit sampling interval Δt  

The input is usually arranged as a tensor:

- Shape: `(T, C)` or `(1, C, T)`  
  - `T`: sequence length (number of time steps)  
  - `C`: number of channels (e.g., voltage, current, temperature)

### 2.2 Preprocessing Pipeline

To match training conditions, the following preprocessing steps are recommended:

1. **Fixed sampling interval**  
   - Resample raw data to the same sampling period used during training (e.g., 1 s or 10 s).

2. **Sliding window construction**  
   - Maintain a rolling window of length `T` (e.g., 100–300 time steps).
   - At each inference step, feed the latest `T` samples into USFFNet.

3. **Normalization**  
   - Normalize voltage, current (C-rate), and temperature using the same statistics as in training (mean/variance or min/max).
   - Ensure **SoC operating window alignment** (e.g., [10 %, 90 %]) if such constraints were applied during training.

4. **Frequency-domain features**  
   - The frequency branch in USFFNet typically uses FFT on the time-domain sequences.
   - You can either:
     - Compute FFT **inside** the model (preferred for cleaner deployment), or  
     - Precompute FFT **outside** and feed both time- and frequency-domain features.

In an embedded setting, it is common to **pre-normalize** and possibly precompute FFT on the device, then pass the processed window to the inference engine.

---

## 3. Software Stack

A typical deployment pipeline involves the following components:

1. **Training framework**  
   - PyTorch (or similar deep learning framework).

2. **Export format**  
   - ONNX as an intermediate representation for portability.

3. **Target runtime / inference engine**
   - Embedded Linux / SoC:
     - ONNX Runtime, TensorRT, TVM, or similar.
   - MCU / bare-metal:
     - TFLite Micro, CMSIS-NN, or a custom fixed-point C inference engine.

---

## 4. Model Export

### 4.1 PyTorch → ONNX

Example export script

```python
import torch

from usffnet import USFFNet 

# 1. Load trained model
model = USFFNet(...)
model.load_state_dict(torch.load("usffnet_checkpoint.pth", map_location="cpu"))
model.eval()

# 2. Create dummy input (shape must match your deployment input)
# Example: batch_size=1, channels=3 (V, I, T), sequence length=T
dummy_input = torch.randn(1, 3, T)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "usffnet.onnx",
    input_names=["input"],
    output_names=["gamma", "nu", "alpha", "beta"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch_size"}}
)
```

After exporting:

- Use **ONNX Runtime** on a PC to verify numerical correctness.
- Optionally run a batch of test samples and compare outputs vs. PyTorch.

### 4.2 ONNX → Backend-specific Format

Depending on the target:

- **TensorRT (Jetson / embedded GPU / some SoCs)**
  - Convert `usffnet.onnx` to a TensorRT engine (`.plan` file).
  - Optimize for FP16 or INT8 if the hardware supports it.
- **TFLite / TFLite Micro (MCU, mobile, some SoCs)**
  - Convert ONNX/PyTorch to TFLite using supported toolchains.
  - Use **post-training quantization** (INT8 / INT16) for memory and speed benefits.
  - For MCUs, convert the TFLite model into a C array and use TFLite Micro.
- **CMSIS-NN / custom C**
  - Map each layer (Conv1D, activation, linear, etc.) to available kernels.
  - Embed weights as constant arrays in flash.

------

## 5. Quantization

Quantization helps reduce memory footprint and improve latency while maintaining accuracy.

### 5.1 Recommended Strategy

1. **Post-training quantization** (PTQ):
   - Use INT8 or INT16 weights and activations.
   - Calibrate with a representative dataset (sequences of voltage/current/temperature).
2. **Per-channel weight quantization**:
   - Recommended for convolution layers to preserve accuracy.
3. **Evaluation**:
   - Compare:
     - SoC RMSE / MAE
     - Coverage of uncertainty intervals (if evidential head is kept in quantized pipeline)
   - Ensure performance degradation is acceptable for your application.

### 5.2 Considerations for Uncertainty

- If you keep the **evidential head** in the quantized model:
  - Verify that NIG parameters (γ, ν, α, β) remain numerically stable.
  - Check that prediction intervals still have reasonable coverage.
- In resource-constrained systems, one option is:
  - Use USFFNet for SoC + a simplified uncertainty metric (e.g., approximate variance), or
  - Run the full evidential head in FP32 on slightly more capable SoCs.

------

## 6. Target Platforms and Integration

### 6.1 Embedded Linux SoC (e.g., Jetson-class devices, ARM SoCs)

**Stack example:**

- OS: Ubuntu / embedded Linux
- Runtime: ONNX Runtime or TensorRT
- Optionally packaged in a Docker container for reproducible deployment

**Workflow:**

1. Copy the `usffnet.onnx` (or TensorRT engine) and normalization parameters to the device.
2. Implement an inference service:
   - Collect and preprocess voltage/current/temperature data.
   - Maintain a sliding window and feed it into USFFNet at 1–10 Hz.
   - Post-process outputs to obtain SoC and uncertainty bands.
3. Integrate with the BMS supervisor:
   - Use SoC mean and uncertainty to drive energy management and safety decisions.

### 6.2 Automotive-grade MCU (e.g., ARM Cortex-M / Cortex-R)

**Stack example:**

- Runtime: Bare-metal or RTOS (e.g., FreeRTOS)
- Inference engine: TFLite Micro / CMSIS-NN / custom C

**Workflow:**

1. Convert the trained model to a **quantized TFLite** model.
2. Convert the TFLite model into a C array, or use a supported build system to compile it into firmware.
3. Implement:
   - A small inference wrapper (e.g., `USFFNet_Infer()`).
   - Input preprocessing and sliding-window buffering.
4. Integrate `USFFNet_Infer()` into the BMS control loop:
   - Typical SoC update rate: 1–10 Hz.
   - Ensure the measured worst-case execution time (WCET) is below your control period.

------

## 7. Real-time Integration in BMS

### 7.1 Sampling and Windowing

- Choose a sampling interval Δt consistent with training (e.g., 1–10 s).
- Maintain a buffer of length `T` for each channel.
- On each control tick:
  1. Append new measurements.
  2. Drop oldest ones.
  3. Construct a normalized input tensor and pass it to USFFNet.

### 7.2 Using the Model Outputs

USFFNet outputs NIG parameters (γ, ν, α, β), from which you can derive:

- **Mean SoC estimate** (γ)
- **Predictive variance / interval** (from ν, α, β)

Example usage:

- Use γ as the primary SoC estimate.
- Compute a prediction interval (e.g., 95 % CI) and:
  - Trigger alarms if uncertainty exceeds a threshold.
  - Use conservative bounds in critical decision-making (e.g., limiting charge/discharge at low SoC when uncertainty is high).

### 7.3 Safety and Monitoring

- Log SoC predictions and uncertainty for offline analysis.
- Monitor for:
  - Sudden rises in uncertainty (possible sensor faults or abnormal operating conditions).
  - Persistent bias between estimated SoC and reference (if available).

------

## 8. Reproducibility Checklist

To reproduce deployment-related results, it is helpful to document:

- **Model configuration**:
  - Network architecture (layers, channels, kernel sizes)
  - Hyperparameters (e.g., λ for evidential regularization, learning rate, epochs)
- **Preprocessing**:
  - Normalization statistics for V, I, T
  - Window length `T` and sampling interval Δt
  - SoC operating window (e.g., [10 %, 90 %])
- **Quantization settings**:
  - Bit width (INT8/INT16)
  - Calibration dataset
- **Target hardware**:
  - MCU / SoC model and clock frequency
  - Available RAM and flash
  - Measured inference latency and CPU load at target update rate

------

## 9. Summary

USFFNet is a compact, dual-domain neural network tailored for **real-time SoC estimation in BMS** with **uncertainty quantification**. With:

- ≈ 2.38 × 10⁴ parameters
- ≈ 93 KB model size (FP32)
- < 0.3 MB runtime memory
- On the order of 10⁵ FLOPs per inference

USFFNet is well suited for deployment on **resource-constrained embedded hardware**. By following the export, quantization, and integration steps outlined in this guide, practitioners can deploy USFFNet on both **MCU-based** and **embedded Linux-based** platforms to obtain reliable, uncertainty-aware SoC estimates in real-world battery management systems.
