import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import string

def generate_dataset(n_samples=1000):
    """Generates an extended dataset of comments with toxicity labels.

    Args:
        n_samples: The number of samples to generate.

    Returns:
        A Pandas DataFrame containing the comment data.
    """
    # Create a list of base toxic words
    base_toxic_words = [
        'stupid', 'idiot', 'loser', 'hate', 'kill', 'moron', 'dumb', 'pathetic', 'annoying',
        'ignorant', 'disgusting', 'worthless', 'despicable', 'vile', 'arrogant', 'selfish',
        'cruel', 'obnoxious', 'repugnant', 'hateful'
    ]

    # Create a list of non-toxic words
    non_toxic_words = [
        'happy', 'love', 'joy', 'peace', 'friend', 'kindness', 'compassion', 'empathy', 'gentle',
        'caring', 'positive', 'uplifting', 'honest', 'sincere', 'authentic', 'trustworthy', 'loyal',
        'supportive', 'considerate', 'respectful', 'generous', 'forgiving', 'patient', 'tolerant',
        'open-minded', 'harmonious', 'intelligent', 'creative', 'talented', 'wise', 'ambitious',
        'balanced', 'healthy', 'graceful', 'playful', 'fun-loving', 'cheerful', 'optimistic',
        'lively', 'vivacious', 'benevolent', 'charitable', 'humane', 'ethical', 'honorable', 'modest',
        'humble', 'content', 'mindful', 'spiritual', 'tranquil', 'serene', 'hopeful', 'motivational',
        'educational', 'enlightening', 'captivating', 'engaging', 'entertaining', 'enjoyable',
        'heartwarming', 'comforting', 'refreshing', 'uplifting'
    ]

    # Extend the list of toxic words with synonyms and antonyms
    extended_toxic_words = base_toxic_words.copy()
    for word in base_toxic_words:
        synonyms = set()
        antonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())

        # Add synonyms and antonyms
        extended_toxic_words.extend(list(synonyms))
        non_toxic_words.extend(list(antonyms))

    # Create a list of comments and their respective toxicities
    comments = []
    toxicities = []

    for i in range(n_samples):
        # Choose a random toxicity label
        toxicity = np.random.choice(['toxic', 'non-toxic'])

        # Generate a comment
        comment = ''
        for j in range(10):
            # Choose a random word
            word = np.random.choice(extended_toxic_words if toxicity == 'toxic' else non_toxic_words)

            # Add the word to the comment
            comment += word + ' '

        # Add the comment and its toxicity to the respective lists
        comments.append(comment.strip())  # Remove trailing space
        toxicities.append(toxicity)

    # Create a Pandas DataFrame containing the comment data
    df = pd.DataFrame({'comment': comments, 'toxicity': toxicities})

    return df

# Generate an extended dataset of 1000 comments
df = generate_dataset(n_samples=1000)

# Save the dataset to a CSV file
df.to_csv('extended_toxic_comments_dataset.csv', index=False)
