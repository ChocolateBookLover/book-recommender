import streamlit as st                     # Streamlit = makes our app into a website
import requests                           # Lets us talk to websites/APIs like Open Library
from sklearn.feature_extraction.text import TfidfVectorizer   # Turns text into numbers for AI
from sklearn.metrics.pairwise import cosine_similarity       # Compares how similar texts are

# ----------------------------------------------------
# Page Title
# ----------------------------------------------------
st.title("📚 Tanvika's AI Book Recommender")

# ----------------------------------------------------
# Language Selector (starts blank on purpose)
# ----------------------------------------------------
language = st.selectbox(
    "🌍 Select the language you want your books in:",
    ["English", "French", "German", "Spanish", "Italian", "Russian"],
    index=None,                            # index=None = start with nothing selected
    placeholder="Choose a language..."
)

# This maps what the user sees → what Open Library expects
language_map = {
    "English": "eng",
    "French": "fre",
    "German": "ger",
    "Spanish": "spa",
    "Italian": "ita",
    "Russian": "rus"
}

# ----------------------------------------------------
# Age Range Selector (also starts blank)
# ----------------------------------------------------
age_range = st.selectbox(
    "🎯 Select the age range for the book:",
    ["Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=None,
    placeholder="Choose an age range..."
)

# ----------------------------------------------------
# User Description Input
# ----------------------------------------------------
user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, emotional coming-of-age):"
)

# ----------------------------------------------------
# Start Over Button
# ----------------------------------------------------
if st.button("🔄 Start Again"):
    st.rerun()    # Reloads the whole app and clears everything

# ----------------------------------------------------
# Function: Fetch books from Open Library
# ----------------------------------------------------
def fetch_books(query, language_code):
    # This is the website we're asking for book data
    url = f"https://openlibrary.org/search.json?q={query}"

    response = requests.get(url)           # Send request to Open Library
    data = response.json()                 # Convert response into Python dictionary

    books = []

    # We loop through results and collect only useful info
    for doc in data.get("docs", [])[:30]:
        # Skip books that aren't in the selected language
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


# ----------------------------------------------------
# AI + Recommendation Logic
# ----------------------------------------------------
if user_input and language and age_range:     # Only runs when ALL inputs are filled

    language_code = language_map[language]    # Convert English → eng, French → fre, etc

    books = fetch_books(user_input, language_code)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        # This is where the AI part starts
        # We compare what the user typed to each book description

        corpus = [user_input] + [book["description"] for book in books]

        vectorizer = TfidfVectorizer()        # Turns text into math vectors
        vectors = vectorizer.fit_transform(corpus)

        similarities = cosine_similarity(vectors[0], vectors[1:])[0]

        ranked_books = sorted(
            zip(similarities, books),
            reverse=True,
            key=lambda x: x[0]
        )

        st.subheader("✨ Recommended Books")

        # Show top 3 books
        for score, book in ranked_books[:3]:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(book["description"])

            # Show book cover if available
            if book["cover_id"]:
                cover_url = f"https://covers.openlibrary.org/b/id/{book['cover_id']}-L.jpg"
                st.image(cover_url, width=150)

            st.write(f"🤖 AI Match Score: {round(score, 2)}")
            st.write("---")
