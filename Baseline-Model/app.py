import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

# Load the pre-trained model and vectorizer
with open('model_rf_tfidf.pkl', 'rb') as model_file:
    model_rf_tfidf = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)  

# Confirm the type of the loaded vectorizer
print(f"Loaded vectorizer type: {type(vectorizer)}")

# Define emoji for sentiments
emoji_mapping = {
    'Neutral': '😐',  # Blank face for Neutral
    'Positive': '😊',  # Happy face for Positive
    'Negative': '😢'   # Sad face for Negative
}

def predict_sentiment(text):
    # Check if vectorizer is of the correct type
    if isinstance(vectorizer, TfidfVectorizer):
        # Transform the input text using the loaded vectorizer
        text_transformed = vectorizer.transform([text])
        
        # Predict using the Random Forest model
        prediction = model_rf_tfidf.predict(text_transformed)
        probabilities = model_rf_tfidf.predict_proba(text_transformed)[0]
        
        # Map numerical predictions back to sentiment labels
        sentiment_mapping = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
        sentiment_label = sentiment_mapping[prediction[0]]
        
        return sentiment_label, probabilities
    else:
        raise TypeError("Loaded vectorizer is not of type TfidfVectorizer")

def main():

    # Title and description
    st.title("Sentiment Analysis Prediction")
    st.write("Analyze the sentiment of any text you input! This tool will predict whether the sentiment is Neutral, Positive, or Negative.")

    # Input from the user
    user_input = st.text_area("Enter Text Here:")

    if st.button("Predict Sentiment"):
        if user_input:
            try:
                sentiment, probabilities = predict_sentiment(user_input)
                
                # Display the predicted sentiment with an emoji
                emoji = emoji_mapping[sentiment]
                st.subheader(f"The predicted sentiment is: **{sentiment} {emoji}**")
                
                # Display the probabilities for each sentiment
                st.write("### Sentiment Probabilities:")
                sentiment_labels = ['Neutral', 'Positive', 'Negative']
                for i, label in enumerate(sentiment_labels):
                    emoji = emoji_mapping[label]
                    st.write(f"- **{emoji} {label}**: {probabilities[i] * 100:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to predict.")

if __name__ == "__main__":
    main()
