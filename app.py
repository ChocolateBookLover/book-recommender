# ===============================
# IMPORTS
# ===============================
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# APP TITLE
# ===============================
st.title("📚 Tanvika's AI Book Recommender")
# Big title for the app

# ===============================
# AGE RANGE SELECTOR
# ===============================
age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=0
)
# Dropdown for age range; blank by default

# ===============================
# USER DESCRIPTION INPUT
# ===============================
user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)
# User types description of the book they want

# ===============================
# GOOGLE BOOKS FUNCTION
# ===============================
def fetch_books_googlebooks(query, max_results=30):
    """
    Fetch books from Google Books API.
    Returns a list of dictionaries with:
        - title
        - author
        - description
        - cover_id (URL to thumbnail)
        - published_date
    """
    books = []

    # Google Books API URL
    # You can add &key=YOUR_API_KEY if you have one, otherwise it works publicly
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"

    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching data from Google Books API")
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
# OPEN LIBRARY FUNCTION (Fallback)
# ===============================
def fetch_books_openlibrary(query):
    """
    Fetch books from Open Library.
    Returns a list of dictionaries with:
        - title
        - author
        - description
        - cover_id
    """
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    data = response.json()

    books = []

    for doc in data.get("docs", [])[:30]:
        description = (
            doc.get("first_sentence", ["No description available"])[0]
            if isinstance(doc.get("first_sentence"), list)
            else doc.get("first_sentence", "No description available")
        )

        books.append({
            "title": doc.get("title", "Unknown title"),
            "author": ", ".join(doc.get("author_name", ["Unknown author"])),
            "description": description,
            "cover_id": f"https://covers.openlibrary.org/b/id/{doc.get('cover_i')}-L.jpg" if doc.get("cover_i") else None,
            "published_date": doc.get("first_publish_year", "N/A")
        })

    return books

# ===============================
# SOURCE SELECTOR
# ===============================
source = st.radio(
    "Choose book source:",
    ("Google Books", "Open Library")
)
# Lets user choose which source to fetch books from

# ===============================
# AI LOGIC: TF-IDF + COSINE SIMILARITY
# ===============================
if user_input:
    # Fetch books based on selected source
    if source == "Google Books":
        books = fetch_books_googlebooks(user_input)
    else:
        books = fetch_books_openlibrary(user_input)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        # Prepare texts for AI: user input first, then all book descriptions
        corpus = [user_input] + [book["description"] for book in books]

        # Convert text into numbers (TF-IDF)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        # Compare user input with each book
        similarities = cosine_similarity(vectors[0], vectors[1:])[0]

        # Pair similarity score with book dictionary
        ranked_books = sorted(
            zip(similarities, books),  # zip combines similarity + book
            reverse=True,              # highest similarity first
            key=lambda x: x[0]
        )

        # ===============================
        # DISPLAY RESULTS
        # ===============================
        st.subheader("✨ Recommended Books")
        for score, book in ranked_books[:3]:  # Top 3
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(f"**Published Date:** {book['published_date']}")
            st.write(f"**Age Range:** {age_range if age_range else 'N/A'}")
            st.write(book["description"])
            st.write(f"🤖 AI Match Score: {round(score, 2)}")

            # Display cover if available
            if book["cover_id"]:
                st.image(book["cover_id"], width=150)

            st.write("---")  # divider

# ===============================
# START AGAIN BUTTON
# ===============================
if st.button("🔄 Start Again"):
    st.rerun()
