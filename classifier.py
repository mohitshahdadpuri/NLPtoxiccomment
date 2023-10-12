import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('all')
import string
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# OpenAI chatbot API
df = pd.read_csv('extended_toxic_comments_dataset.csv')

def preprocess_comment(comment):
    """Preprocesses a comment for classification.

    Args:
        comment: A string containing the comment to preprocess.

    Returns:
        A list of preprocessed tokens.
    """
    # Lowercase the comment
    comment = comment.lower()

    # Remove punctuation
    comment = comment.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    # Tokenize the comment
    tokens = word_tokenize(comment)

    # Return the preprocessed tokens
    return tokens

def extract_features(comment):
    """Extract features from a preprocessed comment.

    Args:
        comment: A list of preprocessed tokens.

    Returns:
        A dictionary of feature-value pairs.
    """
    return {word: True for word in comment}

def train_model(df):
    """Trains a model to classify comments into toxic or non-toxic.

    Args:
        df: A Pandas DataFrame containing the comment data.

    Returns:
        A trained model.
    """
    # Prepare the data for classification
    labeled_data = [(extract_features(preprocess_comment(comment)), toxicity) for comment, toxicity in zip(df['comment'], df['toxicity'])]

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(labeled_data)

    # Return the trained classifier
    return classifier

def classify_comment(tokens, classifier):
    """Classifies a comment into one of two categories: toxic or non-toxic.

    Args:
        tokens: A list of preprocessed tokens.
        classifier: A trained model.

    Returns:
        A string containing the predicted toxicity category.
    """
    # Classify the comment
    category = classifier.classify(extract_features(tokens))

    # Return the predicted toxicity category
    return category

st.title('Toxic Comment Classifier')

# Get the comment from the user
comment = st.text_input('Enter a comment to classify:', '')

# Preprocess the comment
tokens = preprocess_comment(comment)

# Load the trained model
classifier = train_model(df)

# Classify the comment
category = classify_comment(tokens, classifier)

# Display the predicted toxicity category
st.write('Predicted toxicity category:', category)
