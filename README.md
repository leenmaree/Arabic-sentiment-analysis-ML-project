# Arabic Sentiment Analysis

## Overview
This project is an Arabic Sentiment Analysis system that classifies tweets into sentiment classes using machine learning. It includes preprocessing, feature extraction, model training, and evaluation.

---

## Features
- Arabic text cleaning and preprocessing  
- TF-IDF + handcrafted features  
- Models:
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - Neural Network (MLP)  
- Hyperparameter tuning  
- Handling class imbalance using SMOTE  
- Evaluation (Accuracy, Precision, Recall, F1-score)  
- Confusion matrix visualization  
- Interactive prediction mode  

---

## Dataset
- Input file: `.txt` (tab-separated)  
- Format:

```txt
Tweet    Class
هذا المنتج رائع    POS
الخدمة سيئة جدا   NEG
