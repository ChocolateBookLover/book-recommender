import streamlit as st
import requests

st.title("Tanvika's Book Recommender")

user_description = st.text_input("Describe the kind of book you want:")

if user_description:
    st.write("Searching for:", user_description)

    query = user_description.replace(" ", "+")
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=10"

    response = requests.get(url)
    st.write("API status:", response.status_code)

    data = response.json()

    books = []
    for item in data.get("items", []):
        info = item.get("volumeInfo", {})
        books.append({
            "title": info.get("title", "No title"),
            "description": info.get("description", "No description available"),
            "authors": ", ".join(info.get("authors", ["Unknown"])),
            "thumbnail": info.get("imageLinks", {}).get("thumbnail")
        })

    if not books:
        st.write("No books found.")
    else:
        displayed_books = books[:5]

        for i, book in enumerate(displayed_books):
            st.markdown(f"### {book['title']} by {book['authors']}")
            if book["thumbnail"]:
                st.image(book["thumbnail"], width=120)
            st.write(book["description"])

        if st.button("Start Again"):
            st.experimental_rerun()
