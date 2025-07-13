# 📧 Naive Bayes Spam Classifier

This project implements a **Naive Bayes classifier from scratch using Python** to distinguish between spam and ham (non-spam) messages. It uses the classic SMS Spam Collection Dataset and evaluates performance by computing accuracy manually.

---

## 🚀 Getting Started

1. Clone the repository:
  ```bash
  git clone https://github.com/boogen/ml-course.git
  cd ml-course/bayes_spam_classifier
  ```


2. Install dependencies in a virtual environment
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip3 install -r requirements.txt
  ```
  
3. Run the script:
  ```bash
  python3 model.py
  ```

---

## 🧠 Algorithm Overview

- **Type**: Multinomial Naive Bayes
- **Features**: Bag-of-words representation
- **Smoothing**: Laplace (add-one)
- **Training**: Manual parsing of messages and word counts by class
- **Prediction**: Log-probability comparison between `spam` and `ham`

---

## 📊 Dataset

- Source: `data/spam.csv`
- Format: Two columns — `label` (spam/ham) and `text` (message body)
- Preprocessing: Tokenization, lowercase normalization

---

## 📈 Results

The model was evaluated on a stratified 80/20 train/test split.

```
Accuracy: 0.9847 (956/971)
```

*(Note: Result may vary slightly depending on split seed)*

---

## 🧑‍💻 Author

Created by Marcin Bugala as a hands-on exercise in building machine learning algorithms from scratch.  
This project is intended for learning and exploration