import json
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load FAQ Data
try:
    with open("faqs.json", "r") as file:
        faq_data = json.load(file)
except Exception as e:
    print("Error loading faqs.json:", e)
    faq_data = []

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

def get_answer(user_question):
    if not questions:
        return "FAQ data not loaded."
    processed_questions = [preprocess(q) for q in questions]
    processed_user = preprocess(user_question)
    all_questions = processed_questions + [processed_user]
    vectorizer = TfidfVectorizer().fit_transform(all_questions)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])
    idx = np.argmax(cosine_sim)
    if cosine_sim[0][idx] < 0.3:
        return "Sorry, I don't understand your question."
    return answers[idx]

if __name__ == "__main__":
    print("Hello! Ask me anything about our FAQs. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = get_answer(user_input)
        print("Bot:", response)
