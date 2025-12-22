# 🗣️ Word-Level Lip Reading Using Visual Information

A comparative study of **5 deep learning architectures** for Visual Speech Recognition (VSR), developed for the **DSE312: Computer Vision** course at **IISER Bhopal**. This project explores solutions to the "Viseme Ambiguity" problem using Spatiotemporal convolutions, GRUs, and context-aware aggregation (which we might reference as attention in the future).

![Project Banner](plots/comparison_plot.png)

## 📄 Abstract & Problem Statement
Standard speech recognition fails in noisy environments or when audio is unavailable. Lip reading offers a solution but faces two core challenges:
1.  **Viseme Ambiguity:** Many phonemes look identical (e.g., /p/, /b/, /m/), making words like "pat", "bat", and "mat" hard to distinguish.
2.  **Spatiotemporal Complexity:** The model must capture rapid, subtle changes in lip shape over time.

## 📊 Model Evolution & Results

We implemented and benchmarked **5 distinct architectures** to evaluate the impact of 3D Convolutions, Color (RGB), and Attention mechanisms.

| Model | Architecture | Input | Accuracy | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | 2D CNN | Gray | **FAILED** | Model was too simple; did not converge. |
| **Model 2** | 3D CNN + GRU | Gray | **62.52%** | Success. Spatiotemporal (3D) + Sequential (GRU) works. |
| **Model 3** | 3D CNN + GRU | RGB | **67.97%** | Adding color provided more information and improved accuracy. |
| **Model 4** | **3D CNN + GRU + Att** | **RGB** | **88.14%** | **Best Model.** Focusing on specific frames (Attention) yields the best results. |
| **Model 5** | 3D CNN + GRU + Att | Gray (200) | **75.29%** | Architecture scales well to larger vocabularies. |

> **Performance Note:** Our best model achieved an ROC AUC of **0.99** (Macro-average), with perfect classification (AUC 1.00) for several classes.

## 🧠 Methodology

### 1. Preprocessing
* **Face Detection:** Utilized **dlib** face landmarks tool.
* **Cropping:** Extracted the specific mouth region from video frames to remove background noise.

### 2. Architectures Implemented
* **3D CNNs:** Used to extract spatiotemporal features from the video volume, capturing motion dynamics better than 2D CNNs.
* **GRU (Gated Recurrent Units):** Processed the sequence of features extracted by the CNNs to model time dependencies.
* **Attention:** Applied a linear softmax layer to assign weights to hidden GRU states, allowing the model to focus on the most relevant frames for classification.

### 3. Datasets
The models were trained and evaluated using data subsets from:
* **MIRACL-VC1:** 15 speakers, 10 words, 3000 instances.
* **Lip Reading in the Wild (LRW):** 500 words, ~500k instances.
* **Lip Reading Sentences (LRS):** Pre-training performed on large sentence corpuses.

## 📂 Project Structure

```text
Lip-Reading-Project/
├── dataloaders/       # Custom dataset classes for MIRACL/LRW
├── models/            # Implementations of 3D-CNN, GRU, and Attention
├── plots/             # ROC curves and Loss graphs
├── utils/             # Helper functions (dlib preprocessing, etc.)
├── train.py           # Main training script
└── test.py            # Evaluation script
