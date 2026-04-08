# Image Sequence Position Classification using Deep Learning

A deep learning project that tackles the challenging task of predicting an image's position (1-5) within a sequential story using Convolutional Neural Networks (CNNs) built with PyTorch.

## 📋 Table of Contents

- Introduction
- Problem Statement
- Solution Approach
- Project Structure
- Installation
- Usage
- Experiments
- Results
- Key Findings

## 🎯 Introduction

This project explores whether **static visual features alone** can encode temporal sequence information in narrative stories. Unlike text-based approaches that can rely on explicit temporal markers (e.g., "first," "then," "finally"), this research investigates if CNNs can learn to identify an image's position in a story sequence purely from visual cues.

**Academic Context**: This is a Deep Neural Networks reassessment task involving comprehensive ablation studies to understand the impact of:

- Regularization techniques (dropout)
- Model capacity (filter counts)
- Architectural choices (kernel sizes, batch normalization)

## 🔍 Problem Statement

### Task Definition

Given a single image from a 5-frame story sequence, predict its position (1, 2, 3, 4, or 5) in the narrative.

### Challenges

1. **Temporal Ambiguity**: Static images lack explicit temporal markers
2. **Context Dependency**: Understanding position often requires seeing multiple frames
3. **Visual Similarity**: Similar actions/scenes may appear at different positions
4. **Limited Data**: Training on 300 stories (1,500 images) constrains model complexity

### Dataset

- **Source**: HuggingFace `daniel3303/StoryReasoning` dataset
- **Training**: 300 stories × 5 images = 1,500 images
- **Split**: 80% train (1,200 images) / 20% validation (300 images)
- **Classes**: 5 positions (perfectly balanced - 300 samples per class)
- **Image Size**: Resized to 32×64 pixels for efficient training

## 💡 Solution Approach

### Architecture: 3-Layer CNN

```
Input (3×32×64)
    ↓
Conv1 (3→32 filters, 3×3) → ReLU → MaxPool
    ↓
Conv2 (32→64 filters, 3×3) → ReLU → MaxPool
    ↓
Conv3 (64→128 filters, 3×3) → ReLU → AdaptiveAvgPool
    ↓
Flatten → FC (4096→128) → ReLU → Dropout → FC (128→5)
    ↓
Output (5 classes)
```

### Key Design Choices

1. **Preprocessing**:
   - Histogram equalization for brightness normalization
   - Resize to 32×64 (preserves aspect ratio, reduces computation)

2. **Model Configuration**:
   - **Base filters**: 32 (doubles at each layer)
   - **Kernel size**: 3×3 (smaller receptive field)
   - **Dropout**: 0.3 (moderate regularization)
   - **Batch normalization**: Optional (tested in ablations)

3. **Training**:
   - **Optimizer**: Adam (lr=0.001)
   - **Loss**: CrossEntropyLoss (multi-class classification)
   - **Epochs**: 5 (sufficient for convergence on this task)
   - **Batch size**: 32

4. **Evaluation Metrics**:
   - Training/Validation Loss
   - Training/Validation Accuracy
   - Loss Gap (overfitting detection)

## 📁 Project Structure

```
krish/
├── main.py                              # Main execution pipeline
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore patterns
├── README.md                            # Project documentation
├── DNN_Reassessment_Complete.ipynb      # Jupyter notebook (original)
│
├── src/                                 # Source modules
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Configuration & hyperparameters
│   ├── data_loader.py                   # Dataset loading & preprocessing
│   ├── model.py                         # CNN architecture
│   ├── train.py                         # Training & evaluation functions
│   └── visualization.py                 # Plotting utilities
│
├── Results/                             # Generated results (auto-created)
│   ├── dropout_curves.png               # Experiment 1 curves
│   ├── no_dropout_curves.png            # Experiment 2 curves
│   ├── kernel_5_curves.png              # Experiment 3 curves
│   ├── filters_64_curves.png            # Experiment 4 curves
│   ├── batchnorm_curves.png             # Experiment 5 curves
│   ├── comprehensive_comparison.png     # All experiments comparison
│   ├── results_table.csv                # Summary table
│   └── Analysis_Answers.txt             # Viva preparation answers
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

**Create virtual environment** (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

**Install dependencies**:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start

Run all experiments with a single command:

```bash
python main.py
```

This will:

1. Load the StoryReasoning dataset
2. Create train/validation splits
3. Run 5 ablation experiments
4. Generate individual training curves
5. Create comprehensive comparison plots
6. Save results table to CSV

### Using the Notebook

For interactive exploration, open `DNN_Reassessment_Complete.ipynb` in Jupyter or Google Colab and run cells sequentially.

## 🔬 Experiments

Five ablation studies were conducted to understand model behavior:

| Experiment         | Modification                 | Purpose                                |
| ------------------ | ---------------------------- | -------------------------------------- |
| **1. Dropout=0.3** | Baseline with regularization | Standard configuration                 |
| **2. No Dropout**  | dropout=0.0                  | Test impact of removing regularization |
| **3. Kernel=5**    | kernel=5 vs 3                | Larger receptive field                 |
| **4. Filters=64**  | filters=64 vs 32             | Double model capacity                  |
| **5. BatchNorm**   | Add batch normalization      | Gradient stabilization                 |

## 📊 Results

### Performance Summary

| Experiment     | Train Acc | Val Acc       | Train Loss | Val Loss | Loss Gap     |
| -------------- | --------- | ------------- | ---------- | -------- | ------------ |
| 1. Dropout=0.3 | 19.58%    | **22.00%** ⭐ | 1.610      | 1.609    | -0.001       |
| 2. No Dropout  | 22.67%    | 15.00%        | 1.609      | 1.615    | 0.006        |
| 3. Kernel=5    | 20.50%    | 16.33%        | 1.609      | 1.615    | 0.007        |
| 4. Filters=64  | 21.67%    | 19.67%        | 1.608      | 1.613    | 0.005        |
| 5. BatchNorm   | 36.50%    | 16.67%        | 1.465      | 1.698    | **0.234** ⚠️ |

**Random Baseline**: 20% (5-class classification)

### Visualizations

All plots are automatically saved to the `Results/` directory:

- Individual experiment curves (loss & accuracy)
- Comprehensive 6-panel comparison
- Final accuracy bar charts
- Overfitting detection plots

## 🔑 Key Findings

### 1. Task Difficulty

- **All models perform near random baseline (~20%)**, indicating the task is extremely challenging
- Static images lack temporal/sequential markers needed to determine position
- Visual features alone are insufficient without broader context

### 2. Best Performance

- **Dropout=0.3** achieved highest validation accuracy (22%)
- Modest regularization helps with limited data (300 stories)
- Minimal improvement over random suggests fundamental task limitations

### 3. Overfitting Detection

- **BatchNorm showed clear overfitting**: 36.5% train vs 16.7% val accuracy
- Loss gap of 0.234 indicates model memorizing training patterns
- Other experiments showed minimal overfitting (gaps < 0.01)

### 4. Model Capacity

- **Increasing filters to 64 did not help** (19.67% vs 22% baseline)
- Limited data (1,500 images) cannot support larger models
- More parameters → fitting noise rather than patterns

### 5. Architectural Choices

- **Kernel size 5 showed no benefit** over kernel size 3
- Larger receptive fields don't capture additional useful features
- Task requires semantic understanding, not just visual patterns

---

**Note**: This project demonstrates that predicting image position from static visual features alone is fundamentally challenging. The near-random performance across all experiments suggests that temporal sequence understanding requires either multi-frame context, explicit temporal markers, or significantly more training data.
