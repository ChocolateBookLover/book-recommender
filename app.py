import streamlit as st       # Streamlit library for web apps
import requests               # Requests library to fetch API data

# --- Page title ---
st.title("Tanvika's Book Recommender")

# --- Start Again button ---
# This button reloads the app from scratch
if st.button("Start Again"):
    st.experimental_rerun()   # Clears all inputs and displayed books

# --- User input ---
# Text input box for the user to describe what they want
user_description = st.text_input("Describe the kind of book you want:")

# --- Only run the book fetching if user typed something ---
if user_description:
    # Replace spaces with "+" for the API query
    query = user_description.replace(" ", "+")
    
    # Google Books API URL
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=10"
    
    # Send GET request to API
    response = requests.get(url)
    data = response.json()   # Convert JSON response to Python dictionary

    # --- Extract books from API data ---
    books = []
    for item in data.get("items", []):   # Safely get "items", default empty list
        info = item["volumeInfo"]        # Book info dictionary
        books.append({
            "title": info.get("title", "No title"),
            "description": info.get("description", "No description available"),
            "authors": ", ".join(info.get("authors", ["Unknown"])),   # Join authors list
            "thumbnail": info.get("imageLinks", {}).get("thumbnail", None)
        })

    displayed_books = books[:5]

    # Loop through each displayed book
    for i, book in enumerate(displayed_books):
        st.markdown(f"### {book['title']} by {book['authors']}")   # Bold title and author
        if book['thumbnail']:
            st.image(book['thumbnail'], width=120)                  # Show book cover
        st.write(book["description"])                               # Show description

        # Create a button for each book
        if st.button(f"🔄 Already Read {i}"):
            # Replace current book with next one in the API list
            if len(books) > 5 + i:        # Check if more books are available
                displayed_books[i] = books[5 + i]
                st.experimental_rerun()   # Refresh page to show updated book
