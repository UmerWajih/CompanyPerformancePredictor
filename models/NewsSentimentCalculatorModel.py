import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split


# dataset example
data = {
    'article': [
        "E-commerce industry is growing rapidly.",
        "The appliance market faces major supply chain issues.",
        "Global e-commerce growth is expected to increase.",
        "The supply chain for electronics is facing disruptions.",
        "Online shopping platforms see growth during holiday seasons."
    ],
    'sentiment': [1, -1, 1, -1, 1]  # 1 = Positive, -1 = Negative
}

# Convert to DataFrame
df = pd.DataFrame(data)

# 2. Text Preprocessing
max_words = 1000  # Max number of words to keep in the tokenizer
max_len = 20  # Max length of each sequence

tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['article'])

# Convert texts to sequences of integers
X = tokenizer.texts_to_sequences(df['article'])

# Pad sequences to ensure uniform length
X = pad_sequences(X, maxlen=max_len)

# Sentiment labels
y = np.array(df['sentiment'])

# 3. Build RNN Model (LSTM)
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=50, input_length=max_len))  # Embedding layer
model.add(LSTM(units=64, return_sequences=False))  # LSTM layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='tanh'))  # Output layer (using tanh to scale the sentiment score between -1 and 1)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 4. Train the model
model.fit(X, y, epochs=10, batch_size=2)

# 5. Sentiment Prediction Function
def get_sentiment_score(article):
    sequence = tokenizer.texts_to_sequences([article])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    sentiment_score = model.predict(padded_sequence)[0][0]
    return sentiment_score

# Test the model with a dummy article
article = "The e-commerce market has witnessed exponential growth."
predicted_score = get_sentiment_score(article)
print(f"Predicted Sentiment Score for '{article}': {predicted_score}")

# 6. Feedback Loop Simulation
feedback_data = []

# Simulating feedback loop (assuming some predictions are incorrect)
def feedback_loop(article, predicted_score, actual_score):
    if abs(predicted_score - actual_score) > 0.2:  # Threshold for feedback
        print(f"Feedback: Adjusting model for article '{article}' due to discrepancy.")
        feedback_data.append({
            "article": article,
            "predicted_score": predicted_score,
            "actual_score": actual_score
        })

# Example of feedback loop: Actual scores are randomly generated (for testing)
for article in df['article']:
    predicted_score = get_sentiment_score(article)
    actual_score = random.uniform(-1, 1)  # Simulating actual sentiment score
    feedback_loop(article, predicted_score, actual_score)

# 7. Retraining the Model with Feedback (if applicable)
def retrain_model_with_feedback():
    if feedback_data:
        # Update the dataset with the feedback (corrected data)
        feedback_df = pd.DataFrame(feedback_data)
        X_feedback = tokenizer.texts_to_sequences(feedback_df['article'])
        X_feedback = pad_sequences(X_feedback, maxlen=max_len)
        y_feedback = np.array(feedback_df['actual_score'])
        
        # Retrain model with feedback data
        model.fit(X_feedback, y_feedback, epochs=5, batch_size=2)
        print("Model retrained with new feedback.")

# Retrain if there is feedback data
retrain_model_with_feedback()


model.save('news_sentiment_analysis_rnn.h5')

