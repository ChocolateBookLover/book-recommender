# ===============================
# IMPORTS – what we need
# ===============================

import streamlit as st
# Streamlit lets Python run like a website: buttons, inputs, images, text

import requests
# Requests lets Python talk to websites / APIs (Open Library in this case)

from sklearn.feature_extraction.text import TfidfVectorizer
# Converts text (book descriptions) into numbers so AI can compare them

from sklearn.metrics.pairwise import cosine_similarity
# Calculates similarity between user input and book descriptions numerically


# ===============================
# APP TITLE
# ===============================

st.title("📚 Tanvika's AI Book Recommender (Open Library Edition)")
# Big title at the top


# ===============================
# START AGAIN BUTTON
# ===============================

if st.button("🔄 Start Again"):
    st.experimental_rerun()
    # Clears all inputs and displayed books


# ===============================
# AGE RANGE SELECTOR
# ===============================

age_range = st.selectbox(
    "Select age range:",
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
# User types what they want


# ===============================
# FUNCTION TO FETCH BOOKS FROM OPEN LIBRARY
# ===============================

def fetch_books(query):
    """
    query = what user typed
    Returns a list of books with:
        - title
        - author
        - cover id (to show image)
    """
    # Open Library search URL
    url = f"https://openlibrary.org/search.json?q={query}"

    # Send GET request to Open Library
    response = requests.get(url)
    data = response.json()  # Convert JSON response to Python dictionary

    books = []

    # Loop through the first 30 results
    for doc in data.get("docs", [])[:30]:
        # Some books have a 'first_sentence' list; handle both cases
        description = (
            doc.get("first_sentence", ["No description available"])[0]
            if isinstance(doc.get("first_sentence"), list)
            else doc.get("first_sentence", "No description available")
        )

        # Age info is tricky: Open Library doesn't provide it, so we skip
        # For now, we can assume all books fit the age range user wants

        books.append({
            "title": doc.get("title", "Unknown title"),
            "author": ", ".join(doc.get("author_name", ["Unknown author"])),
            "description": description,
            "cover_id": doc.get("cover_i", None)
        })

    return books


# ===============================
# AI LOGIC: TF-IDF + COSINE SIMILARITY
# ===============================

if user_input:
    # Fetch books using the Open Library API
    books = fetch_books(user_input)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        # Prepare texts for AI
        # First element is user input, rest are book descriptions
        corpus = [user_input] + [book["description"] for book in books]

        # Convert text into numerical vectors using TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)
        # fit_transform = learns word importance & converts text to numbers

        # Compare user input with each book
        similarities = cosine_similarity(vectors[0], vectors[1:])[0]
        # returns a list of similarity scores

        # Pair each similarity score with its corresponding book
        ranked_books = sorted(
            zip(similarities, books),  # zip combines similarity + book
            reverse=True,              # highest similarity first
            key=lambda x: x[0]         # sort by similarity score
        )

        # ===============================
        # DISPLAY RESULTS
        # ===============================

        st.subheader("✨ Recommended Books")

        # Show top 3
        for score, book in ranked_books[:3]:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(book["description"])
            st.write(f"🤖 AI Match Score: {round(score, 2)}")

            # If Open Library has a cover image, display it
            if book["cover_id"]:
                cover_url = f"https://covers.openlibrary.org/b/id/{book['cover_id']}-L.jpg"
                st.image(cover_url, width=150)

            st.write("---")  # divider between books
