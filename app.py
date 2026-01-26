# ===============================
# IMPORTS
# ===============================

import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# LOAD GOOGLE BOOKS API KEY FROM SECRETS
# ===============================

try:
    GOOGLE_BOOKS_API = "AIzaSyDFzZtFbcDYxx2YMWl75174pcGsd2jwnKo"
except KeyError:
    st.error(
        "❌ Google Books API key not found!\n"
        "Add it to Streamlit Secrets as:\n\n"
        "GOOGLE_BOOKS_API = 'YOUR_API_KEY'"
    )
    st.stop()

# ===============================
# APP TITLE
# ===============================

st.title("Tanvika's AI Book Recommender")

# ===============================
# AGE RANGE SELECTOR
# ===============================

age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=0
)

# ===============================
# USER INPUT
# ===============================

user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)

# ===============================
# KEYWORD EXTRACTION
# ===============================

# Common words to filter out for better search
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
    'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'as', 'if', 'because', 'until', 'while', 'like', 'want',
    'wants', 'kind', 'type', 'book', 'books', 'story', 'stories', 'read',
    'something', 'anything', 'everything', 'nothing', 'someone', 'anyone'
}

def extract_keywords(text):
    """Extract meaningful keywords from conversational text."""
    import re
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Filter out stop words and short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(keywords) if keywords else text

# ===============================
# GOOGLE BOOKS API FUNCTION
# ===============================

def fetch_books_googlebooks(query, max_results=40):
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
        try:
            st.json(response.json())  # Show the API error if available
        except:
            pass
        return books

    data = response.json()

    for item in data.get("items", []):
        volume_info = item.get("volumeInfo", {})
        description = volume_info.get("description")

        # Skip books without descriptions (can't do AI matching)
        if not description:
            continue

        title = volume_info.get("title", "Unknown Title")
        authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
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
# AI LOGIC
# ===============================

if user_input:
    # Extract keywords for better Google Books search
    search_query = extract_keywords(user_input)
    books = fetch_books_googlebooks(search_query)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        # Combine title and description for better AI matching
        corpus = [user_input] + [f"{book['title']} {book['description']}" for book in books]
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
