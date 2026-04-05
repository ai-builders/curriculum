# Requirements Document

## Introduction

This document specifies the requirements for `notebooks/01_ml_what_2026.ipynb`, an English-language 2026 update of the AI Builders curriculum notebook "What is Machine Learning?" (`notebooks/01_ml_what.ipynb`). The original notebook is in Thai and covers AI/ML/DL definitions, rule-based vs ML systems, the training loop, metrics, train/validation/test splits, and hands-on examples (image, text, tabular). The 2026 update introduces three key changes: (1) LLM Systems as a third paradigm alongside rule-based and ML, (2) modern evaluation beyond clean metrics, and (3) a DSPy prompt-optimization demo that reframes "prompts as weights."

## Glossary

- **Notebook**: The Jupyter notebook file `notebooks/01_ml_what_2026.ipynb`
- **Colab_Environment**: Google Colab runtime where the Notebook executes
- **Conceptual_Section**: A markdown cell or group of markdown cells that explain a concept
- **Code_Demo**: A code cell or group of code cells that demonstrate a concept with runnable code
- **ASCII_Diagram**: A text-based diagram rendered in a markdown cell, keeping all visuals self-contained in the notebook (no external image files)
- **Training_Loop_Diagram**: An ASCII_Diagram of the cycle: Inputs → Weights → Predictions → Loss → Gradients → Optimizer → updated Weights
- **DSPy_Demo**: A hands-on Code_Demo using the DSPy library to optimize prompts, treating prompts as weights
- **LLM_System**: An AI system built on top of a pre-trained LLM, using prompts (zero-shot, few-shot, or optimized) to perform tasks — closer to software engineering than traditional ML
- **Three_Paradigms_Section**: A Conceptual_Section presenting three categories of AI systems: Rule-based, ML, and LLM Systems

## Requirements

### Requirement 1: Notebook Structure and Colab Compatibility

**User Story:** As a student, I want to open the notebook in Google Colab and run all cells sequentially, so that I can follow the lesson without environment setup issues.

#### Acceptance Criteria

1. THE Notebook SHALL be a valid `.ipynb` file located at `notebooks/01_ml_what_2026.ipynb`
2. THE Notebook SHALL include a Colab badge markdown cell linking to the notebook on the ai-builders/curriculum GitHub repository
3. THE Notebook SHALL include a package installation cell that installs all required dependencies for the Colab_Environment
4. THE Notebook SHALL also support local execution via `uv` for dependency management — a markdown cell SHALL document how to set up and run the notebook locally (e.g., `uv venv && uv pip install -r requirements.txt` or inline `uv pip install` commands)
5. THE Notebook SHALL be written entirely in English
6. WHEN a student runs all cells sequentially in the Colab_Environment, THE Notebook SHALL execute without import errors or missing dependency errors
7. WHEN a developer runs the notebook locally on a MacBook Pro (Apple Silicon, no GPU), all cells SHALL execute successfully, though training cells may run slower than on Colab GPU
8. ALL diagrams and visuals SHALL be ASCII_Diagrams rendered in markdown cells — no external image file dependencies

### Requirement 2: AI / ML / DL Definitions

**User Story:** As a student, I want to understand the definitions of AI, ML, and DL and how they relate to each other, so that I can use these terms correctly.

#### Acceptance Criteria

1. THE Conceptual_Section SHALL define Artificial Intelligence (AI) as systems that think or act intelligently
2. THE Conceptual_Section SHALL define Machine Learning (ML) as a subset of AI where rules are learned from data
3. THE Conceptual_Section SHALL define Deep Learning (DL) as ML techniques using multi-layer neural networks
4. THE Conceptual_Section SHALL define Large Language Models (LLMs) as a subset of DL — large-scale neural networks trained on text to predict the next token, noting that while LLMs are DL models, the way they are used (as LLM Systems via prompts) represents a different paradigm covered in Requirement 3
5. THE Conceptual_Section SHALL include an ASCII_Diagram showing the nested relationship of AI ⊃ ML ⊃ DL ⊃ LLMs

### Requirement 3: Three Paradigms of AI Systems (Rule-based, ML, LLM Systems)

**User Story:** As a student, I want to understand the three major paradigms for building AI systems, so that I can recognize when each approach is appropriate.

#### Acceptance Criteria

1. THE Three_Paradigms_Section SHALL describe Rule-based Systems where humans define explicit rules that transform inputs into outputs
2. THE Three_Paradigms_Section SHALL describe Machine Learning Systems where rules (models) are learned from historical input-output pairs
3. THE Three_Paradigms_Section SHALL describe LLM Systems where a pre-trained LLM performs tasks via prompts (zero-shot, few-shot, or optimized) without training a task-specific model from scratch
4. THE Three_Paradigms_Section SHALL explain that LLM Systems represent a paradigm shift closer to software engineering than traditional ML — you compose and prompt rather than collect data and train
5. THE Three_Paradigms_Section SHALL include an ASCII_Diagram or markdown table comparing the three paradigms (who defines the rules, what data is needed, how the system is built)

### Requirement 4: The Training Loop

**User Story:** As a student, I want to understand how ML models learn through the training loop, so that I can grasp the core mechanism behind model training.

#### Acceptance Criteria

1. THE Conceptual_Section SHALL explain the Training_Loop_Diagram components: Inputs, Labels, Weights, Predictions, Loss Function, Gradients, Optimizer
2. THE Conceptual_Section SHALL explain the concepts of batch, epoch, and iteration
3. THE Conceptual_Section SHALL include an ASCII_Diagram of the training loop showing the flow from Inputs through Weights, Predictions, Loss, Gradients, Optimizer, and back to updated Weights

### Requirement 5: Evaluation — Metrics, Splits, and Beyond

**User Story:** As a student, I want to understand how to measure model performance across both traditional ML and LLM systems, so that I can evaluate any AI system appropriately.

#### Acceptance Criteria

1. THE Conceptual_Section SHALL explain the difference between Loss and Metric
2. THE Conceptual_Section SHALL provide examples of common metrics for classification (accuracy, precision, recall, F1) and regression (MSE, MAE)
3. THE Conceptual_Section SHALL explain the purpose of train, validation, and test splits
4. THE Conceptual_Section SHALL explain why the same data must not appear in multiple splits
5. THE Conceptual_Section SHALL include an ASCII_Diagram of train/validation/test splits
6. THE Conceptual_Section SHALL pivot from traditional metrics to the challenge of evaluating free-text outputs, using the HotpotQA demo as a concrete example ("exact match didn't always work, so we used token F1 — but what if even that isn't enough?")
7. THE Conceptual_Section SHALL briefly name-drop modern LLM evaluation approaches: LLM-as-judge, human evaluation, rubric-based scoring — without going into implementation detail
8. THE Conceptual_Section SHALL include a forward-reference to `notebooks/03a_metrics_and_baselines.ipynb` as the lesson where evaluation will be covered in depth (including 2026 updates for LLM evaluation)

### Requirement 6: Hands-On Traditional ML Demos

**User Story:** As a student, I want to see traditional ML training examples with real data, so that I can connect the training loop theory to practice.

#### Acceptance Criteria — 6a: Image Classification (FoodyDudy)

1. THE Code_Demo SHALL train an image classification model on the FoodyDudy Thai food dataset (48 classes) from the original notebook, using fastai/ResNet or an equivalent high-level framework
2. THE Code_Demo SHALL show the key training loop components (data loading, pretrained model, fine-tuning, metrics) in action
3. THE Code_Demo SHALL print or display accuracy on a validation set after training
4. WHEN a student runs the Code_Demo cells in the Colab_Environment, THE Code_Demo SHALL produce visible output (metrics and sample predictions)

#### Acceptance Criteria — 6b: Neural Network Regression with Curve-Fitting Visualization

1. THE Code_Demo SHALL use PyTorch Lightning to build a neural network from scratch, demonstrating the training loop components via LightningModule (model, training_step, configure_optimizers)
2. THE Code_Demo SHALL use non-linear real or synthetic data where a straight line clearly fails to fit
3. THE Code_Demo SHALL first train a single-layer linear model and plot its predicted line overlaid on the data, showing the poor fit
4. THE Code_Demo SHALL then train a multi-layer neural network on the same data and plot its predicted curve, showing how adding layers enables non-linear fitting
5. THE Code_Demo SHALL plot the model's predicted curve overlaid on the data points at multiple points during training, showing how the curve progressively fits the data distribution as weights are updated
6. THE Code_Demo SHALL display the loss value at each plotted step so students can see loss decreasing as the fit improves
7. WHEN a student runs the Code_Demo cells in the Colab_Environment, THE Code_Demo SHALL produce a series of plots (or an animation) showing the curve evolving over training iterations for both the linear and non-linear models

### Requirement 7: Hands-On LLM System Demo — Manual Prompt Engineering then Automatic Optimization

**User Story:** As a student, I want to first try writing and iterating on prompts myself, and then see how DSPy automates that same process, so that I can understand prompt optimization as the LLM-era equivalent of training weights.

#### Acceptance Criteria — 7a: Manual Prompt Engineering

1. THE Code_Demo SHALL use the HotpotQA dataset (distractor split, loaded via HuggingFace `datasets`) as the task — multi-hop question answering with short free-text answers
2. THE Code_Demo SHALL load Qwen3-8B (quantized to 4-bit) as the LLM via HuggingFace Transformers — the same model used in 7b
3. THE Code_Demo SHALL provide a simple evaluation function that scores the prompt's output on a held-out set using token-level F1 (the standard HotpotQA metric), so students can see their prompt's metric score
4. THE Code_Demo SHALL encourage students to iterate: try different prompts, add examples, rephrase instructions, and re-run to see if the metric improves
5. THE Conceptual_Section SHALL frame this as "manual weight tuning" — the prompt is the weight, and the student is the optimizer

#### Acceptance Criteria — 7b: Automatic Prompt Optimization with DSPy

1. THE DSPy_Demo SHALL use the DSPy library to define the same HotpotQA task from 7a
2. THE DSPy_Demo SHALL use Qwen3-8B (quantized to 4-bit) as the LLM, loaded locally via HuggingFace Transformers — this model fits in Colab T4 GPU (16GB VRAM) and can also run on a MacBook Pro via CPU/MPS
3. THE DSPy_Demo SHALL define a DSPy module with a signature specifying inputs and outputs
4. THE DSPy_Demo SHALL use a DSPy optimizer (e.g., BootstrapFewShot or MIPROv2) to optimize the prompt automatically
5. THE DSPy_Demo SHALL show the before-optimization and after-optimization performance on the same metric used in 7a, so students can compare their manual result to the automated one
6. THE DSPy_Demo SHALL include a Conceptual_Section that explicitly maps DSPy concepts to training loop concepts (prompt ↔ weights, optimizer ↔ optimizer, metric ↔ loss/metric, training examples ↔ training data)
7. WHEN a student runs the DSPy_Demo cells in the Colab_Environment, THE DSPy_Demo SHALL produce visible output showing the optimization progress and final metric

### Requirement 8: Vocabulary Review

**User Story:** As a student, I want a glossary of key terms at the end of the notebook, so that I can review and reinforce what I learned.

#### Acceptance Criteria

1. THE Notebook SHALL include a vocabulary review section listing key terms and their definitions
2. THE vocabulary review section SHALL include both traditional ML terms (Inputs, Labels, Weights, Loss, Gradients, Optimizer, Epoch, Batch, Metric, Train/Validation/Test Set) and new terms introduced in the 2026 update (LLM, LLM System, Prompt Optimization, LLM-as-Judge)

### Requirement 9: Reflection Questions

**User Story:** As a student, I want thought-provoking questions at the end of the notebook, so that I can reflect on what I learned and apply it to my own project ideas.

#### Acceptance Criteria

1. THE Notebook SHALL include a reflection questions section with at least four questions
2. THE reflection questions SHALL cover: (a) identifying real-world examples of the three AI paradigms, (b) when to choose traditional ML vs. LLM-based approaches, (c) how to evaluate an LLM-based system for a given task, (d) comparing the experience of manual prompt engineering vs. automatic optimization — what did DSPy find that you didn't?
