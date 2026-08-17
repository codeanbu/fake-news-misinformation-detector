# 📰 AI-Powered Fake News & Misinformation Detector

An end-to-end NLP application that detects whether a news article is **Fake or Real** using a fine-tuned **DistilBERT Transformer** model.

The application supports direct article input, TXT files, CSV batch prediction, PDF extraction, and news URL extraction. It also provides prediction confidence, probability visualization, and model explainability.

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://fake-news-misinformation-detector-hhfvkmpg6sreyqqrlf9x8j.streamlit.app

🤗 **Hugging Face Model:**  
https://huggingface.co/anbucode/fake-news-distilbert

---

## 📌 Project Overview

Fake news and online misinformation can spread rapidly through digital platforms.

This project uses Natural Language Processing and Transformer-based deep learning to classify news articles into:

- 🔴 Fake News
- 🟢 Real News

The project follows a complete machine learning workflow:

**Data → Preprocessing → Model Training → Evaluation → Explainability → Deployment**

---

## 🤖 Model

### Fine-Tuned DistilBERT

The primary model used in this project is:

**DistilBERT**

DistilBERT is a lightweight Transformer architecture derived from BERT. It provides strong NLP performance while requiring fewer computational resources than the original BERT model.

The model was fine-tuned specifically for binary fake-news classification.

### Classification

| Class | Label |
|------|------|
| 🔴 Fake | 0 |
| 🟢 Real | 1 |

---

## 📊 Dataset

The dataset contains:

**44,889 news articles**

The dataset contains two classes:

- Fake News
- Real News

### Dataset Distribution

| Class | Articles |
|------|---------:|
| Fake | 23,472 |
| Real | 21,417 |
| **Total** | **44,889** |

---

## 🧠 Machine Learning Approaches

Multiple machine learning approaches were explored before selecting the Transformer-based model.

### Traditional Machine Learning

The following models were evaluated using TF-IDF features:

- Random Forest
- Linear SVM
- Logistic Regression
- Naive Bayes

### Transformer Model

The final system uses:

**TF Text → Tokenization → DistilBERT → Classification Head → Prediction**

---

## 📈 Model Performance

### Traditional ML Results

| Model | Accuracy |
|------|---------:|
| Random Forest | 99.78% |
| Linear SVM | 99.57% |
| Logistic Regression | 99.03% |
| Naive Bayes | 93.74% |

### DistilBERT

The fine-tuned DistilBERT model achieved:

| Metric | Score |
|------|------:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

> Note: These results are based on the evaluation setup used during model development. Real-world performance can vary when the model encounters unseen sources, topics, writing styles, or deliberately manipulated content.

---

## 🔍 Application Features

### 📝 1. Text Classification

Users can paste a complete news article into the application.

The system returns:

- Fake / Real prediction
- Confidence score
- Fake probability
- Real probability

---

### 📂 2. File Upload

The application supports:

- `.txt`
- `.csv`
- `.pdf`

TXT files can be directly classified.

CSV files can be used for batch prediction.

---

### 📊 3. Batch Prediction

CSV files containing a `text` column can be classified in batch.

The application generates:

- Prediction
- Confidence score

Users can download the results as a new CSV file.

---

### 🌐 4. News URL Analysis

Users can provide a news article URL.

The application extracts the article content and sends it to the trained classifier.

---

### 📑 5. PDF Article Extraction

PDF documents can be processed by extracting their text before classification.

---

### 📊 6. Probability Visualization

The application provides an interactive Plotly visualization showing:

- Fake probability
- Real probability

---

### 🧠 7. Model Explainability

LIME-based explainability is included to identify words that contributed to the model's prediction.

This helps users understand why the model classified an article as Fake or Real.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User Input     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
         Text Input       News URL          File Upload
                                             │
                                      ┌──────┴──────┐
                                      │             │
                                     TXT           PDF/CSV
                                      │             │
                                      └──────┬──────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Text Extraction     │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ DistilBERT Tokenizer │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Fine-Tuned DistilBERT│
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Softmax Probabilities│
                                  └──────────┬──────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                         🔴 Fake                        🟢 Real
                              │
                              ▼
                    Confidence Visualization
                              │
                              ▼
<<<<<<< HEAD
                       LIME Explainability
=======
                       LIME Explainability
>>>>>>> f6c9170 (Add professional project documentation)
