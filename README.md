NLP-Based Review Classifier

This project is a machine learning application designed to automatically classify movie reviews as either positive or negative. It leverages Natural Language Processing (NLP) techniques to analyze text data and predict sentiment. The system is built using Python, scikit-learn for the core machine learning model, and Flask to provide a prediction API.

Features
Sentiment Classification: The core functionality of the project is to classify movie reviews into one of two categories: positive (pos) or negative (neg).

TF-IDF Vectorization: Text data is converted into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency), which helps the model understand the importance of words in the context of the reviews.

Logistic Regression Model: A Logistic Regression model is trained on a large dataset of movie reviews to learn the patterns associated with positive and negative sentiments.

Flask API: A lightweight web API is provided, allowing external applications to send a review text and receive a sentiment prediction.

Project Files
main.py: This script is the heart of the machine learning pipeline. It downloads and processes the NLTK movie_reviews dataset, splits it into training and testing sets, trains the Logistic Regression model, and evaluates its performance. Finally, it saves the trained model (sentiment_model.pkl) and the TF-IDF vectorizer (tfidf_vectorizer.pkl) for use in the API.

app.py: This Flask application serves as the prediction server. It loads the pre-trained model and vectorizer and creates a /predict API endpoint. This endpoint takes a JSON payload with a text field and returns a JSON response containing the sentiment prediction.

Usage
1. Setup

Clone the repository and install the required libraries.

Bash

# Install the necessary packages
pip install Flask scikit-learn nltk joblib
2. Model Training

Run the main.py script to train the model and save the required files (sentiment_model.pkl and tfidf_vectorizer.pkl).

Bash

python main.py
3. Running the API

Start the Flask application to launch the prediction server.

Bash

python app.py
