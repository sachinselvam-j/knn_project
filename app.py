import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Spam Detection - KNN", layout="centered")
st.title("📩 Spam Detection using KNN")

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    return df[["label", "text"]]

df = load_data()

# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # ham=0, spam=1

X = df["text"]
y = df["label"]

# ----------------------------
# Train model (NO PICKLE)
# ----------------------------
@st.cache_resource
def train_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X, y)
    return model

model = train_model()

st.success("✅ KNN model trained successfully")

# ----------------------------
# User input
# ----------------------------
st.subheader("Enter a message")

user_input = st.text_area("Message text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        prediction = model.predict([user_input])[0]
        result = le.inverse_transform([prediction])[0]

        if result == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
