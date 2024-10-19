import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    text = text.lower()
    words = [word for word in text.split() if word.isalnum()]
    return ' '.join(words)

def load_data(path):
    data = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            label = 'spam' if filename.startswith('spm') else 'ham'
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                preprocessed_content = preprocess_text(content)
                data.append([preprocessed_content, label])
    return pd.DataFrame(data, columns=['text', 'label'])

train_path = r'static\train_test_mails\train-mails' 
train_df = load_data(train_path)

test_path = r'static\train_test_mails\test-mails' 
test_df = load_data(test_path)

train_df['label'] = train_df['label'].map({'spam': 1, 'ham': 0})
test_df['label'] = test_df['label'].map({'spam': 1, 'ham': 0})

X_train = train_df['text']
X_test = test_df['text']

y_train = train_df['label']
y_test = test_df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=False),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

print("\nAccuracy Scores:")
best_model = max(results, key=results.get)
best_accuracy = results[best_model]

for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.6f}")
print(f'\nBest Model: {best_model} \nAccuracy: {best_accuracy}') 
