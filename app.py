# ===============================
# IMPORTS – Tools our app needs
# ===============================

import streamlit as st          # Streamlit turns Python into a web app
import requests                 # Lets us talk to the Google Books API
from sklearn.feature_extraction.text import TfidfVectorizer   # Converts text into numbers
from sklearn.metrics.pairwise import cosine_similarity        # Compares how similar texts are

# ===============================
# GOOGLE BOOKS API KEY
# ===============================
# For now, put your API key directly here
GOOGLE_BOOKS_API = "YOUR_REAL_GOOGLE_BOOKS_API_KEY"  # <-- REPLACE with your key

# ===============================
# APP TITLE
# ===============================

st.title("Tanvika's AI Book Recommender")

# ===============================
# AGE RANGE SELECTOR (Optional)
# ===============================

age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8–12)", "Teens (13–17)", "Young Adult (18–25)", "Adult (25+)"],
    index=0
)

# ===============================
# USER INPUT – What kind of book do they want?
# ===============================

user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)

# ===============================
# GOOGLE BOOKS API FUNCTION
# ===============================

def fetch_books_googlebooks(query, max_results=30):
    books = []
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&key={GOOGLE_BOOKS_API}"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("❌ Error fetching data from Google Books API")
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
            "des
