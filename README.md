# AI Word Recognition Workstation: Dual-Engine Benchmarking Suite

An advanced, production-grade Deep Learning and Computer Vision workstation built to recognize handwritten words and sequences. This platform features a hybrid **Custom CNN + NLP Joint Probability Decoder** running alongside a state-of-the-art **End-to-End Vision Transformer (TrOCR)** pipeline, wrapped in a high-end, low-light cyber-dark operational dashboard.

---

## ⚡ Key Engineering Features

* **Dual-Input Ingestion Framework:**
  * **Interactive Canvas:** High-resolution live drawing coordinates tracking multi-character stroke inputs.
  * **File Ingestion Pipeline:** Automated grayscale normalization and contrast bitwise-inversion handling external image uploads (`.png`, `.jpg`, `.jpeg`).
* **Dual-Engine AI Execution Architecture:**
  * **Engine 1 (Custom CNN + NLP):** Combines localized OpenCV contour extraction with a 4D-Tensor Convolutional Neural Network. Employs **Top-2 Probabilistic Softmax Candidate Decoding** and a statistical dictionary corpus (`pyspellchecker`) to perform context-aware word correction.
  * **Engine 2 (State-of-the-Art TrOCR):** Leverages `microsoft/trocr-base-handwritten` from the Hugging Face hub. Bypasses classic bounding-box limits by utilizing a **Vision Transformer (ViT)** patch encoder and an autoregressive **RoBERTa** language decoder to interpret continuous or cursive scripts.
* **MLOps Performance Benchmarking Suite:** Features real-time latency metric clocks profiling computation speeds down to the millisecond, exposing the efficiency-vs-accuracy trade-offs between localized networks and global self-attention loops.
* **Pro Cyber-Dark UX:** Styled with obsidian backdrops, semantic glowing canvas containers, and gradient typefaces for an enterprise IDE engineering feel.

---

## 🏗️ Technical Architecture Comparison

| Architectural Dimension | Engine 1: Custom CNN + NLP Segmenter | Engine 2: End-to-End Vision Transformer (TrOCR) |
| :--- | :--- | :--- |
| **Model Type** | 2D Convolutional Neural Network (CNN) | Sequence-to-Sequence Vision-Text Transformer |
| **Input Processing** | OpenCV Connected Component Slicing (Contours) | Grid Patch Tokenization (16x16 windows) |
| **Attention Scope** | Localized feature maps per character box | Global Self-Attention across the complete image |
| **Linguistic Layer** | Joint Softmax Permutations + Edit-Distance Lexicon | Autoregressive Decoder Language Modeling |
| **Typical Latency** | **Fast:** 5 ms - 30 ms (CPU Optimized) | **Heavy:** 0.5 s - 1.5 s (Resource Dense) |
| **Cursive Support** | Low (Requires explicit character spacing) | High (Reads connected script continuously) |

---

## 📂 Repository Directory Layout

```text
📂 handwritten-character-recognition/
├── 📂 data/                          # Cached data directories
│   ├── 📂 raw/                       # Source EMNIST data lines
│   └── 📂 processed/                 # Balanced tensor numpy arrays (.npy)
├── 📂 notebooks/                     # Exploratory research notebooks
├── 📂 reports/                       # Visual data diagnostics and metrics outputs
│   ├── confusion_matrix.png
│   └── transformer_patches_visualized.png
├── 📂 saved_models/                  # Serialized weight assets
│   ├── best_handwritten_model.keras  # Trained 47-Class CNN brain
│   └── training_history.npy
├── 📂 src/                           # Structural source sub-modules
│   ├── 📂 models/                    # Network topology profiles
│   ├── 📂 preprocessing/             # Image normalization matrices
│   ├── predict.py                    # Core character evaluation scripts
│   └── train.py                      # Training loop optimization runs
├── 📂 streamlit_app/                 # Dashboard workspace
│   └── app.py                        # Cyber-dark system orchestration script
├── requirements.txt                  # Core ecosystem dependencies
└── README.md                         # Project documentation profile