import nltk
from nltk.corpus import movie_reviews
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('movie_reviews')
data = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
# with open("data.txt",'w') as f:
#     f.write(str(data))
#     f.close()

random.shuffle(data)
texts = [" ".join(words) for words, labels in data]
labels = [labels for words, labels in data]

# Data Splitting, Vectorization, Model Training, and Evaluation
split_index = int(0.8 * len(texts))
X_train, X_test = texts[:split_index], texts[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# Vectorization and Model Training
vectorizer = TfidfVectorizer(stop_words="english",max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training and Evaluation
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")

joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
print(report)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")