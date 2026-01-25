# ===============================
# IMPORTS – Tools our app needs
# ===============================

import streamlit as st          # Streamlit turns Python into a web app
import requests                # Lets us talk to the Google Books API
from sklearn.feature_extraction.text import TfidfVectorizer   # Converts text into numbers
from sklearn.metrics.pairwise import cosine_similarity        # Compares how similar texts are


# ===============================
# APP TITLE
# ===============================

# Big title at the top of the web app
st.title("📚 Tanvika's AI Book Recommender")


# ===============================
# AGE RANGE SELECTOR (Optional)
# ===============================

# Dropdown menu so the user can choose an age group
age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=0  # Starts with a blank option
)


# ===============================
# USER INPUT – What kind of book do they want?
# ===============================

# Text box where the user describes the book they want
user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)


# ===============================
# GOOGLE BOOKS API FUNCTION
# ===============================

def fetch_books_googlebooks(query, max_results=30):
    """
    This function sends the user's text to the Google Books API
    and gets back real books from the internet.

    It returns a list of dictionaries, where each dictionary represents a book:
    - title
    - author
    - description
    - cover_id (image URL)
    - published_date
    """

    books = []  # This will store all the books we collect

    # Google Books API search URL
    # The user's query is added at the end
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"

    # Send request to Google
    response = requests.get(url)

    # If something went wrong, show an error
    if response.status_code != 200:
        st.error("❌ Error fetching data from Google Books API")
        return books

    # Convert the response into Python data
    data = response.json()

    # Loop through each book Google gives us
    for item in data.get("items", []):

        volume_info = item.get("volumeInfo", {})

        # Get book details safely
        title = volume_info.get("title", "Unknown Title")
        authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
        description = volume_info.get("description", "No description available")
        cover_id = volume_info.get("imageLinks", {}).get("thumbnail", None)
        published_date = volume_info.get("publishedDate", "N/A")

        # Store the book in our list
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

    # Step 1: Get real books from Google
    books = fetch_books_googlebooks(user_input)

    if not books:
        st.warning("⚠️ No books found. Try a different description.")

    else:
        # Step 2: Build a list of text for the AI
        # First item = user's description
        # Then every book's description
        corpus = [user_input] + [book["description"] for book in books]

        # Step 3: Convert text into numbers using TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        # Step 4: Compare user text to every book description
        similarities = cosine_similarity(vectors[0], vectors[1:])[0]

        # Step 5: Combine scores + books and sort them
        ranked_books = sorted(
            zip(similarities, books),   # Pair each score with its book
            reverse=True,               # Highest score first
            key=lambda x: x[0]          # Sort by the similarity score
        )


        # ===============================
        # DISPLAY RESULTS
        # ===============================

        st.subheader("✨ AI Recommended Books")

        # Show the top 3 best matches
        for score, book in ranked_books[:3]:

            st.markdown(f"### 📖 {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(f"**Published Date:** {book['published_date']}")
            st.write(f"**Age Range:** {age_range if age_range else 'N/A'}")
            st.write(book["description"])
            st.write(f"🤖 AI Match Score: {round(score, 2)}")

            # Show cover image if available
            if book["cover_id"]:
                st.image(book["cover_id"], width=150)

            st.write("---")  # Divider between books


# ===============================
# RESET BUTTON
# ===============================

# Lets the user start over
if st.button("🔄 Start Again"):
    st.rerun()
