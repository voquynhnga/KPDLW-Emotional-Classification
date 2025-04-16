# Vietnamese Comment Sentiment Classification (KPDLW-Emotional-Classification)

This project builds a sentiment classification system for Vietnamese comments into 3 classes: Negative, Neutral, and Positive. The model utilizes the PhoBERT architecture and is trained using K-Fold Cross-Validation to enhance robustness. The project also includes a simple Flask web application for real-time predictions.

## Features

*   3-class sentiment classification (Negative, Neutral, Positive) for Vietnamese text.
*   Utilizes the **PhoBERT** model (`vinai/phobert-base`), a powerful Transformer model for Vietnamese.
*   Trained with **Stratified K-Fold Cross-Validation (5 Folds)** for more reliable evaluation and model ensembling.
*   Provides a **Flask Web Application** allowing users to input comments and receive classification results.
*   The web interface displays the predicted label and a probability distribution chart for each class.

## Technology Stack

*   **Language:** Python 3
*   **Deep Learning:** PyTorch, Transformers (Hugging Face)
*   **Web Framework:** Flask
*   **Vietnamese NLP:** Pyvi (ViTokenizer)
*   **Preprocessing:** Gensim (simple_preprocess)
*   **Machine Learning & Data:** Scikit-learn, Pandas, Numpy
*   **Visualization (Training):** Matplotlib, Seaborn
*   **Visualization (Web App):** Chart.js

# Use
* Download and unzip **phobert_sentiment_fold.zip**: [Link](https://drive.google.com/drive/folders/1tmm6rFgs_qLR5ZpgG6m_fn2yOl0qW1r3?usp=sharing)
* pip install -r requirements.txt
* python3 app.py
