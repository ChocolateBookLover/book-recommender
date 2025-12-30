import streamlit as st

st.title("Tanvika's AI Book Recommender")

# User input: describe the kind of book they want
user_description = st.text_input("Describe what kind of book you want:")

# Book list with multiple genres and descriptions
books = [
    {"title": "Harry Potter", "genre": ["fantasy", "magic", "adventure"], "description": "A young wizard learns magic at a school."},
    {"title": "Land of Stories", "genre": ["fantasy", "adventure"], "description": "Two siblings fall into a world of fairy tales."},
    {"title": "Powerless", "genre": ["romance"], "description": "A girl survives in a world where she has no powers."},
    {"title": "Percy Jackson", "genre": ["fantasy", "mythology", "adventure"], "description": "A boy discovers he is the son of a Greek god."},
    {"title": "Better Than The Movies", "genre": ["romance"], "description": "A girl navigates high school and relationships."},
    {"title": "Renegades", "genre": ["sci-fi", "adventure"], "description": "Heroes and villains battle for control of a futuristic city."},
    {"title": "Circe", "genre": ["mythology"], "description": "A retelling of the life of the witch Circe from Greek mythology."},
    {"title": "The Traitor's Game", "genre": ["fantasy"], "description": "A spy story with magic and political intrigue."}
]

# CORE BACKBONE LOGIC (keyword matching)
matches = []
if user_description:
    user_description = user_description.lower()
    for book in books:
        # check if any word in the description matches words in the book description
        if any(word in book["description"].lower() for word in user_description.split()):
            matches.append(book["title"])
    matches = matches[:3]  # limit to top 3

    # Display results
    if matches:
        st.write("Recommended books:")
        for title in matches:
            st.write(title)
        
        # Placeholder for AI integration
        st.write("AI explanation will go here later...")
    else:
        st.write("Sorry, no books match that description.")
