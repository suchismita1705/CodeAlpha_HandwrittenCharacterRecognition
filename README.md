# Handwritten Character Recognition using Deep Learning

An industry-standard computer vision system built from scratch to recognize handwritten alphanumeric characters (digits and letters). This project utilizes a custom Convolutional Neural Network (CNN) trained on the EMNIST Balanced dataset and serves an interactive prediction user interface via a Streamlit web dashboard.

## 📊 Project Architecture & Workflow
1. **Data Acquisition:** Downloads 131,600 images across 47 balanced alphanumeric classes using automated streams.
2. **Exploratory Data Analysis (EDA):** Resolves spatial transpositions and maps numeric label indexes right-side up.
3. **Preprocessing Pipeline:** Minimizes gradient explosions by mapping [0-255] pixels to continuous [0.0, 1.0] scales.
4. **CNN Engine:** Implements structural feature extraction using Conv2D, MaxPooling2D, and Regularization layers.
5. **Streamlit Deployment:** Deploys a user web portal featuring automated canvas color inversion rules.

## 🛠️ Project Directory Layout
```text
handwritten-character-recognition/
├── data/
│   ├── processed/          # Standardized normalized matrix arrays
│   └── raw/                # Cached download dataset blocks
├── notebooks/              # Experimental analytical environments
├── reports/                # Heatmaps and metric text charts
├── saved_models/           # Best compiled network weights (.keras)
├── src/
│   ├── evaluation/         # Model metrics reports generation engine
│   ├── models/             # Custom CNN design blueprints
│   └── training/           # Model weight checkpointers controls
├── streamlit_app/          # Interactive web UI portal dashboard scripts
└── requirements.txt        # Production library tracking manifest