# NeuralTrail
> Find what your model misses.

Neural Trail is an open-source AI debugging laboratory designed to help developers understand how and where their models fail.

Instead of looking only at overall accuracy, Neural Trail turns model debugging into an interactive investigation:

**FIND → TRACE → STRESS → EXPERIMENT → IMPROVE**

## What It Does

### FIND — Model Blind Spots
Evaluate a model and identify incorrect predictions, uncertain samples, and recurring failure patterns.

### TRACE — Neural Network X-Ray
Explore the model's architecture and internal representations through layer-wise activation analysis.

### STRESS — Perturbation Lab
Apply controlled perturbations such as rotation, noise, blur, and occlusion to test model robustness and reproduce failures.

### GEMMA — AI Researcher
Use Gemma as an evidence-driven reasoning layer to interpret observed results, generate hypotheses, and suggest experiments.

### EXPERIMENT — Test the Hypothesis
Turn a hypothesis into an actual experiment and measure its effect against the model baseline.

### IMPROVE — Validate Changes
Evaluate model interventions and compare the original model against a genuinely improved version.

## Model Support

Neural Trail is designed to work with multiple model architectures:

- ResNet18
- MobileNet
- Custom PyTorch `.pth` / `.pt` models

Models are handled through a common adapter interface so the analysis pipeline is not tied to a single architecture.

## Why Neural Trail?

A model can have excellent overall accuracy and still fail systematically under specific conditions.

Neural Trail focuses on the questions that accuracy alone cannot answer:

- Where does the model fail?
- What patterns exist in those failures?
- What happens inside the network?
- Can the failure be reproduced?
- What hypothesis explains the observed behavior?
- Does an intervention actually improve the model?

## Architecture


                ┌─────────────────┐
                │   Model Input   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Model Adapter   │
                │ ResNet / Mobile│
                │ / Custom PyTorch│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Analysis Engine │
                └────────┬────────┘
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
        FIND           TRACE          STRESS
          └──────────────┼──────────────┘
                         ↓
                ┌─────────────────┐
                │ Gemma Researcher│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │   EXPERIMENT    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │     IMPROVE     │
                └─────────────────┘


## Example

Our custom PyTorch CNN demonstrates how Neural Trail can expose robustness failures that are hidden by normal accuracy.

The model achieves approximately **98.75% accuracy** on its normal evaluation set, but controlled perturbation experiments reveal severe degradation under rotation, noise, blur, and occlusion.

This demonstrates the core idea of Neural Trail:

> **High accuracy does not necessarily mean a model is robust.**

## Tech Stack

- Python
- PyTorch
- Torchvision
- React
- TypeScript
- Gemma
- Custom model adapters
- Interactive visualization

## Open Source

Neural Trail is built as an open-source project with the goal of making model debugging more transparent, reproducible, and accessible.

## Team

Built during the Hack Day by:

- Vidisha Jain
- Ankita Lenka
- Ojas Omprakash Karole
- Ananya Pathak

---

**Neural Trail — Find what your model misses.**
