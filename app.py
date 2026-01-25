# ===============================
# IMPORTS – Tools our app needs
# ===============================

import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# GOOGLE BOOKS API KEY
# ===============================
# Replace with your actual API key
GOOGLE_BOOKS_API = "YOUR_REAL_GOOGLE_BOOKS_API_KEY"  # <-- REPLACE THIS

# ===============================
# APP TITLE
# ===============================

st.title("Tanvika's AI Book Recommender")

# ===============================
# AGE RANGE SELECTOR (Optional)
# ===============================

age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=0
)

# ===============================
# USER INPUT – What kind of book do they want?
# ===============================

user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)

# ===============================
# GOOGLE BOOKS API FUNCTION (SAFER)
# ===============================

def fetch_books_googlebooks(query, max_results=30):
    books = []

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "key": GOOGLE_BOOKS_API
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"❌ Error fetching data from Google Books API (status {response.status_code})")
        return books

    data = response.json()

    for item in data.get("items", []):
        volume_info = item.get("volumeInfo", {})
        title = volume_info.get("title", "Unknown Title")
        authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
        description = volume_info.get("description", "No description available")
        cover_id = volume_info.get("imageLinks", {}).get("thumbnail", None)
        published_date = volume_info.get("publishedDate", "N/A")

        books.append({
            "title": title,
            "author": authors,
            "description": description,
            "cover_id": cover_id,
            "published_date": published_date
        })

    return books

# ===============================
# AI LOGIC – Runs after user types something
# ===============================

if user_input:
    books = fetch_books_googlebooks(user_input)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        corpus = [user_input] + [book["description"] for book in books]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(vectors[0], vectors[1:])[0]
        ranked_books = sorted(zip(similarities, books), reverse=True, key=lambda x: x[0])

        st.subheader("AI Recommended Books")

        for score, book in ranked_books[:3]:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(f"**Published Date:** {book['published_date']}")
            st.write(f"**Age Range:** {age_range if age_range else 'N/A'}")
            st.write(book["description"])
            st.write(f"🤖 AI Match Score: {round(score, 2)}")
            if book["cover_id"]:
                st.image(book["cover_id"], width=150)
            st.write("---")

# ===============================
# RESET BUTTON
# ===============================

if st.button("Start Again"):
    st.rerun()
