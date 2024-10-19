import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocess_text(text):
    text = text.lower()
    words = [word for word in text.split() if word.isalnum()]
    return ' '.join(words)

def load_data(path):
    data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            label = 1 if filename.startswith('spm') else 0 
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                preprocessed_content = preprocess_text(content)
                data.append([preprocessed_content, label])
    return pd.DataFrame(data, columns=['text', 'label'])

df = load_data(r'static\train_test_mails\train-mails')
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

model = MultinomialNB()
model.fit(X, df['label'])

with open('model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)