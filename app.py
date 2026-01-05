import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page title

st.title("📚 Tanvika's AI Book Recommender")


# Language Button 


option = st.selectbox
"Select which language you want your books to be in:"
("English, French, German, Russian, Spanish, Italian")
st.write('You selected:', option)

language_map = {
    "English": "eng",
    "French": "fre",
    "German": "ger",
    "Russian": "rus",
    "Spanish": "spa",
    "Italian": "ita"
}

language_code = language_map[option]


# Start Again button

if st.button("🔄 Start Again"):
    st.rerun()


# User input

user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, emotional coming-of-age):"
)


# Fetch books from Open Library

def fetch_books(query, language_code):
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    data = response.json()

    books = []

    for doc in data.get("docs", [])[:30]:
        if language_code not in doc.get("language", []):
            continue

        books.append({
            "title": doc.get("title", "Unknown title"),
            "author": ", ".join(doc.get("author_name", ["Unknown author"])),
            "description": (
                doc.get("first_sentence", ["No description available"])[0]
                if isinstance(doc.get("first_sentence"), list)
                else doc.get("first_sentence", "No description available")
            ),
            "cover_id": doc.get("cover_i")
        })

    return books



# AI logic

if user_input:
   books = fetch_books(user_input, language_code)

if not books:
        st.warning("No books found. Try a different description.")
else:
        # Prepare text for AI
        corpus = [user_input] + [book["description"] for book in books]

        vectorizer = TfidfVectorizer()
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
