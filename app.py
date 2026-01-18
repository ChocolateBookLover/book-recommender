import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Page Title
# =========================
st.title("📚 Tanvika's AI Book Recommender")

# =========================
# Language Selector
# =========================
language_option = st.selectbox(
    "🌍 Select book language:",
    ["English", "French", "German", "Russian", "Spanish", "Italian"]
)

language_map = {
    "English": "eng",
    "French": "fre",
    "German": "ger",
    "Russian": "rus",
    "Spanish": "spa",
    "Italian": "ita"
}

language_code = language_map[language_option]

# =========================
# Age Range Selector
# =========================
age_option = st.selectbox(
    "🎯 Select age range:",
    ["Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"]
)

age_map = {
    "Kids (8–12)": ["children", "kids", "juvenile", "middle grade"],
    "Teens (13–17)": ["teen", "young adult", "ya", "high school"],
    "Young Adult (18–25)": ["young adult", "new adult", "college"],
    "Adult (25+)": ["adult", "mature", "classic", "literary"]
}

age_keywords = age_map[age_option]

# =========================
# Start Again Button
# =========================
if st.button("🔄 Start Again"):
    st.rerun()

# =========================
# User Input
# =========================
user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, emotional coming-of-age):"
)

# =========================
# Fetch Books from Open Library
# =========================
def fetch_books(query, language_code, age_keywords):
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    data = response.json()

    books = []

    for doc in data.get("docs", [])[:40]:
        languages = doc.get("language", [])

        if languages and language_code not in languages:
            continue

        text_blob = " ".join([
            doc.get("title", ""),
            " ".join(doc.get("subject", [])[:10])
        ]).lower()

        if age_keywords and not any(word in text_blob for word in age_keywords):
            continue

        description_parts = []

        if doc.get("first_sentence"):
            if isinstance(doc["first_sentence"], list):
                description_parts.append(doc["first_sentence"][0])
            else:
                description_parts.append(doc["first_sentence"])

        if doc.get("subject"):
            description_parts.append(" ".join(doc["subject"][:6]))

        if not description_parts:
            description_parts.append(doc.get("title", ""))

        books.append({
            "title": doc.get("title", "Unknown title"),
            "author": ", ".join(doc.get("author_name", ["Unknown author"])),
            "description": " ".join(description_parts),
            "cover_id": doc.get("cover_i")
        })

    return books

# =========================
# AI Matching Logic
# =========================
if user_input:
    books = fetch_books(user_input, language_code, age_keywords)

    if not books:
        st.warning("No books found. Try a different description, language, or age range.")
    else:
        corpus = [user_input] + [book["description"] for book in books]

        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(corpus)

        similarities = cosine_similarity(vectors[0], vectors[1:])[0]

        ranked_books = sorted(
            zip(similarities, books),
            reverse=True,
            key=lambda x: x[0]
        )

        st.subheader("✨ Recommended Books")

        for score, book in ranked_books[:3]:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(book["description"])

            if book["cover_id"]:
                cover_url = f"https://covers.openlibrary.org/b/id/{book['cover_id']}-L.jpg"
                st.image(cover_url, width=150)

            st.write(f"🤖 AI Match Score: {round(score, 2)}")
            st.write("---")
