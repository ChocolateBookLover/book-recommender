import streamlit as st

st.title("📚 Tanvika's Book Recommender (Backbone)")

# Button to reset the app
if st.button("Start Again"):
    st.experimental_rerun()  # clears input and results

# User input: describe the kind of book they want
user_description = st.text_input("Describe what kind of book you want:")

# Book list with multiple genres and descriptions
books = [
    {"title": "Harry Potter", "genre": ["fantasy", "adventure"],
     "description": "A young wizard learns magic at a school."},
    {"title": "Land of Stories", "genre": ["fantasy", "adventure"],
     "description": "Two siblings fall into a world of fairy tales."},
    {"title": "Powerless", "genre": ["romance","dystopian", "adventure"],
     "description": "A girl survives in a world where she has no powers."},
    {"title": "Percy Jackson", "genre": ["fantasy", "mythology", "adventure"],
     "description": "A boy discovers he is the son of a Greek god."},
    {"title": "Better Than The Movies", "genre": ["romance", "realistic fiction"],
     "description": "A girl navigates high school and relationships."},
    {"title": "Renegades", "genre": ["sci-fi", "adventure", "romance"],
     "description": "Heroes and villains battle for control of a futuristic city."},
    {"title": "Circe", "genre": ["mythology"],
     "description": "A retelling of the life of the witch Circe from Greek mythology."},
    {"title": "The Traitor's Game", "genre": ["fantasy", "romance", "adventure"],
     "description": "An adventure story with magic and corruption."}
]

matches = []
if user_description:
    user_description = user_description.lower()
    for book in books:
        # match if any word from the user input is in the book description or genre
        if any(word in book["description"].lower() for word in user_description.split()) or \
           any(word in [g.lower() for g in book["genre"]] for word in user_description.split()):
            matches.append(book)
    
    matches = matches[:3]  # limit to top 3 matches

    # Display results
    if matches:
        st.write("Recommended books:")
        for book in matches:
            st.markdown(f"**{book['title']}**")
            st.write(f"*Genres:* {', '.join(book['genre'])}")
            st.write(book["description"])
            st.write("---")
        
        # Placeholder for AI integration
        st.write("AI explanation will go here tomorrow...")
    else:
        st.write("Sorry, no books match that description.")
