import feedparser
import sys
import json
import nltk
import numpy as np
from nltk.corpus import wordnet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load your sentiment analysis model
model = load_model('news_sentiment_analysis_rnn.h5')

# Setup tokenizer (ensure this matches training tokenizer config)
tokenizer = Tokenizer(num_words=1000, lower=True)

# Function to get synonyms for a word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' ').lower())
    return synonyms

# Expand all tags into a flat set with weights
def expand_tags(tag_dict):
    expanded = {}
    for importance, tag_list in tag_dict.items():
        for tag in tag_list:
            synonyms = get_synonyms(tag)
            for synonym in synonyms:
                expanded[synonym] = 4 - importance  # Weight: 1 -> 3, 2 -> 2, 3 -> 1
    return expanded

# Score an article based on tag presence
def score_article(text, expanded_tags):
    text = text.lower()
    score = 0
    for keyword, weight in expanded_tags.items():
        if keyword in text:
            score += weight
    return score

# Predict sentiment from a list of articles
def predict_sentiments(articles):
    sequences = tokenizer.texts_to_sequences(articles)
    padded = pad_sequences(sequences, maxlen=20)
    predictions = model.predict(padded)
    return predictions.flatten()

# Main execution
def main(city, tags_json):
    rss_url = f"https://news.google.com/rss/headlines/section/geo/{city}"
    feed = feedparser.parse(rss_url)

    tags_dict = json.loads(tags_json)
    expanded_tags = expand_tags(tags_dict)

    relevant_articles = []
    original_titles = []

    for entry in feed.entries:
        full_text = f"{entry.title} {entry.summary}"
        tag_score = score_article(full_text, expanded_tags)

        if tag_score >= 3:  # Threshold to filter relevant articles
            relevant_articles.append(full_text)
            original_titles.append(entry.title)

    if not relevant_articles:
        print("No relevant articles found.")
        return

    sentiments = predict_sentiments(relevant_articles)

    print("\n📰 Relevant News Sentiment Scores:")
    for title, score in zip(original_titles, sentiments):
        print(f"Title: {title}\nSentiment Score: {score:.2f}\n")

#Run using the following command
#python news_sentiment.py "Seattle" '{"1": ["ecommerce", "electronics"], "2": ["shopping"], "3": ["discounts"]}'

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python news_sentiment.py <city> <tag_dict_json>")
        print('Example: python news_sentiment.py "Seattle" \'{"1": ["ecommerce", "electronics"], "2": ["shopping"], "3": ["discounts"]}\'')
        sys.exit(1)

    city_name = sys.argv[1]
    tag_json = sys.argv[2]
    main(city_name, tag_json)
