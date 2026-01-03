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

    # --- Fantasy ---
    {
        "title": "Harry Potter",
        "author": "J.K. Rowling",
        "description": "A young wizard attends a magical school, discovers friendship, and battles dark forces."
    },
    {
        "title": "Percy Jackson",
        "author": "Rick Riordan",
        "description": "A boy discovers he is the son of a Greek god and goes on dangerous mythological adventures."
    },
    {
        "title": "The Hobbit",
        "author": "J.R.R. Tolkien",
        "description": "A reluctant hero embarks on a fantasy adventure involving dragons, treasure, and courage."
    },
    {
        "title": "The Land of Stories",
        "author": "Chris Colfer",
        "description": "Two siblings fall into a magical world of fairy tales and legendary characters."
    },
    {
        "title": "Eragon",
        "author": "Christopher Paolini",
        "description": "A farm boy discovers a dragon egg and is pulled into an epic fantasy war."
    },

    # --- Adventure ---
    {
        "title": "Life of Pi",
        "author": "Yann Martel",
        "description": "A boy survives a shipwreck and embarks on a dangerous journey across the ocean."
    },
    {
        "title": "The Maze Runner",
        "author": "James Dashner",
        "description": "Teenagers must survive a deadly maze while uncovering dark secrets."
    },
    {
        "title": "Hatchet",
        "author": "Gary Paulsen",
        "description": "A boy survives alone in the wilderness after a plane crash."
    },
    {
        "title": "Treasure Island",
        "author": "Robert Louis Stevenson",
        "description": "A classic adventure involving pirates, treasure maps, and betrayal."
    },
    {
        "title": "Into the Wild",
        "author": "Jon Krakauer",
        "description": "A true story about a young man who abandons society to explore the wilderness."
    },

    # --- Coming of Age ---
    {
        "title": "The Catcher in the Rye",
        "author": "J.D. Salinger",
        "description": "A teenager struggles with identity, growing up, and feelings of isolation."
    },
    {
        "title": "The Perks of Being a Wallflower",
        "author": "Stephen Chbosky",
        "description": "A shy teenager navigates friendship, trauma, and self-discovery."
    },
    {
        "title": "Looking for Alaska",
        "author": "John Green",
        "description": "A coming-of-age story about love, loss, and personal growth."
    },
    {
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "description": "A young girl learns about justice, empathy, and racism in the American South."
    },
    {
        "title": "Little Women",
        "author": "Louisa May Alcott",
        "description": "Four sisters grow up and learn about love, responsibility, and independence."
    },

    # --- Realistic Fiction ---
    {
        "title": "The Fault in Our Stars",
        "author": "John Green",
        "description": "Two teenagers fall in love while dealing with serious illness."
    },
    {
        "title": "Wonder",
        "author": "R.J. Palacio",
        "description": "A boy with a facial difference navigates school and acceptance."
    },
    {
        "title": "Bridge to Terabithia",
        "author": "Katherine Paterson",
        "description": "Two children create an imaginary world while coping with real-life challenges."
    },
    {
        "title": "Eleanor Oliphant Is Completely Fine",
        "author": "Gail Honeyman",
        "description": "A socially awkward woman learns how to connect with others."
    },
    {
        "title": "The Book Thief",
        "author": "Markus Zusak",
        "description": "A young girl finds comfort in books during World War II."
    },

    # --- Classics ---
    {
        "title": "1984",
        "author": "George Orwell",
        "description": "A dystopian novel about surveillance, control, and loss of freedom."
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "description": "A classic romance exploring love, class, and misunderstandings."
    },
    {
        "title": "Of Mice and Men",
        "author": "John Steinbeck",
        "description": "Two friends struggle to survive during the Great Depression."
    },
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "description": "A tragic story about ambition, love, and the American Dream."
    },
    {
        "title": "Jane Eyre",
        "author": "Charlotte Brontë",
        "description": "A young woman seeks independence and love in a restrictive society."
    },

    # --- Dystopian ---
    {
        "title": "The Hunger Games",
        "author": "Suzanne Collins",
        "description": "A girl fights for survival in a deadly televised competition."
    },
    {
        "title": "Divergent",
        "author": "Veronica Roth",
        "description": "A society divided by personality traits faces rebellion."
    },
    {
        "title": "Fahrenheit 451",
        "author": "Ray Bradbury",
        "description": "A future society where books are banned and burned."
    },
    {
        "title": "The Giver",
        "author": "Lois Lowry",
        "description": "A boy discovers dark truths about his seemingly perfect society."
    },
    {
        "title": "Station Eleven",
        "author": "Emily St. John Mandel",
        "description": "Survivors rebuild culture after a global pandemic."
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
