import streamlit as st
import numpy as np
import nltk
nltk.download('all')
import string

# Define a function to preprocess the comment
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
    stop_words = nltk.corpus.stopwords.words('english')
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = nltk.stem.WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    # Tokenize the comment
    tokens = nltk.word_tokenize(comment)

    # Return the preprocessed tokens
    return tokens

# Define a function to classify the comment
def classify_comment(tokens):
    """Classifies a comment into one of two categories: toxic or non-toxic.

    Args:
        tokens: A list of preprocessed tokens.

    Returns:
        A string containing the predicted toxicity category.
    """

    # Load the toxic word list
    toxic_words = ['stupid', 'idiot', 'loser', 'hate', 'kill']

    # Count the number of toxic words in the comment
    toxic_word_count = 0
    for word in tokens:
        if word in toxic_words:
            toxic_word_count += 1

    # If the number of toxic words is greater than zero, then the comment is toxic. Otherwise, the comment is non-toxic.
    if toxic_word_count > 0:
        category = 'toxic'
    else:
        category = 'non-toxic'

    # Return the predicted toxicity category
    return category

# Create a Streamlit app
st.title('Toxic Comment Classifier')

# Get the comment from the user
comment = st.text_input('Enter a comment to classify:', '')

# Preprocess the comment
tokens = preprocess_comment(comment)

# Classify the comment
category = classify_comment(tokens)

# Display the predicted toxicity category
st.write('Predicted toxicity category:', category)
