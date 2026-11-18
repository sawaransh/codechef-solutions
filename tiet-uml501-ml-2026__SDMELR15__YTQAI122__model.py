import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = 'cleaned_imdb_dataset.csv'
MODEL_PATH = 'sentiment_model.pkl'


def clean_text(text):
  text = str(text).lower()
  return re.sub(r'[^a-z\s]', '', text)


def train_model():
  df = pd.read_csv(DATA_PATH).dropna(subset=['cleaned_review', 'sentiment'])
  X = df['cleaned_review']
  y = df['sentiment']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  vectorizer = CountVectorizer()
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  classifier = MultinomialNB()
  classifier.fit(X_train_vec, y_train)

  preds = classifier.predict(X_test_vec)
  acc = accuracy_score(y_test, preds)
  print(f'Model Accuracy: {acc * 100:.2f}%')

  bundle = {'vectorizer': vectorizer, 'model': classifier}
  with open(MODEL_PATH, 'wb') as out_file:
    pickle.dump(bundle, out_file)


def run_prediction_example():
  with open(MODEL_PATH, 'rb') as in_file:
    bundle = pickle.load(in_file)

  vectorizer = bundle['vectorizer']
  classifier = bundle['model']

  examples = [
      'Amazing acting and a gripping story from start to finish.',
      'Dull plot, terrible pacing, and completely forgettable.',
  ]

  for text in examples:
    vec = vectorizer.transform([clean_text(text)])
    sentiment = classifier.predict(vec)[0]
    print(f'Review: {text}\nPredicted Sentiment: {sentiment}\n')


if __name__ == '__main__':
  train_model()
  run_prediction_example()