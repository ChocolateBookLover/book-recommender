import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Tanvika's AI Book Recommender")

# --- User input ---
user_input = st.text_input(
    "Describe the kind of book you want (example: magical adventure, emotional fantasy):"
)

# --- Your book database ---
books = [
    {
        "title": "Harry Potter",
        "author": "J.K. Rowling",
        "description": "A young wizard attends a magical school and fights dark forces."
    },
    {
        "title": "Percy Jackson",
        "author": "Rick Riordan",
        "description": "A boy discovers he is the son of a Greek god and goes on adventures."
    },
    {
        "title": "Circe",
        "author": "Madeline Miller",
        "description": "A witch from Greek mythology learns independence and strength."
    },
    {
        "title": "Better Than The Movies",
        "author": "Lynn Painter",
        "description": "A lighthearted high school romance full of humor."
    }
]

# --- AI logic: TF-IDF + cosine similarity ---
if user_input:
    # Combine user input with book descriptions
    corpus = [user_input] + [book["description"] for book in books]

    # Convert text to numerical vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    # Compute similarity between user input and each book
    similarities = cosine_similarity(vectors[0], vectors[1:])[0]

    # Pair similarity scores with books
    ranked_books = sorted(
        zip(similarities, books),
        reverse=True,
        key=lambda x: x[0]
    )

    st.subheader("Recommended Books")

    # Show top 3
    for score, book in ranked_books[:3]:
        st.markdown(f"### {book['title']} — {book['author']}")
        st.write(book["description"])
        st.write(f"🤖 AI Match Score: {round(score, 2)}")
        st.write("---")
