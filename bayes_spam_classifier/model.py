import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class NaiveBayesSpamClassifier:
  def __init__(self):
    self.spam_word_counts = defaultdict(int)
    self.ham_word_counts = defaultdict(int)
    self.spam_total_words = 0
    self.ham_total_words = 0
    self.num_spam = 0
    self.num_ham = 0
    self.vocab = set()

  def tokenize(self, text):
    return re.findall(r'\b\w+\b', text.lower())

  def train(self, messages):
    for _, row in messages.iterrows():
        label, text = row['label'], row['text']
        words = self.tokenize(text)
        if label == 'spam':
          self.num_spam += 1
          for word in words:
            self.spam_word_counts[word] += 1
            self.spam_total_words += 1
            self.vocab.add(word)
        else:
          self.num_ham += 1
          for word in words:
            self.ham_word_counts[word] += 1
            self.ham_total_words += 1
            self.vocab.add(word)

  def predict(self, text):
    words = self.tokenize(text)
    log_spam_prob = math.log(self.num_spam / (self.num_spam + self.num_ham))
    log_ham_prob = math.log(self.num_ham / (self.num_spam + self.num_ham))

    for word in words:
      # Laplace smoothing
      spam_word_prob = (self.spam_word_counts[word] + 1) / (self.spam_total_words + len(self.vocab))
      ham_word_prob = (self.ham_word_counts[word] + 1) / (self.ham_total_words + len(self.vocab))

      log_spam_prob += math.log(spam_word_prob)
      log_ham_prob += math.log(ham_word_prob)

    return 'spam' if log_spam_prob > log_ham_prob else 'ham'



data = pd.read_csv('data/spam.csv', encoding="ISO-8859-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Split into train/test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Train the model
model = NaiveBayesSpamClassifier()
model.train(train_data)

test_data['predicted'] = test_data['text'].apply(model.predict)

# Compute accuracy manually
correct = (test_data['predicted'] == test_data['label']).sum()
total = len(test_data)
accuracy = correct / total

print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")