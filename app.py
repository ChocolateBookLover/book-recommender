# ===============================
# IMPORTS
# ===============================

import streamlit as st
import requests
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# LOAD ENVIRONMENT VARIABLES
# ===============================

load_dotenv()

# ===============================
# LOAD API KEYS FROM ENVIRONMENT
# ===============================

GOOGLE_BOOKS_API = os.getenv("GOOGLE_BOOKS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_BOOKS_API:
    st.error(
        "Google Books API key not found!\n\n"
        "Create a `.env` file with:\n"
        "```\nGOOGLE_BOOKS_API_KEY=your_key_here\n```"
    )
    st.stop()

# ===============================
# GEMINI API SETUP (using google-genai SDK)
# ===============================

gemini_available = False
gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_available = True
    except Exception as e:
        st.warning(f"Gemini API not available: {e}")

# ===============================
# APP TITLE
# ===============================

st.title("Tanvika's AI Book Recommender")

# ===============================
# AI METHOD SELECTOR
# ===============================

if gemini_available:
    ai_method = st.radio(
        "Choose AI recommendation method:",
        ["Gemini AI (Smarter)", "TF-IDF (Local)"],
        horizontal=True
    )
else:
    ai_method = "TF-IDF (Local)"
    if not GEMINI_API_KEY:
        st.info("Add GEMINI_API_KEY to .env for smarter AI recommendations")

# ===============================
# AGE RANGE SELECTOR
# ===============================

age_range = st.selectbox(
    "Select age range (optional):",
    ["", "Kids (8-12)", "Teens (13-17)", "Young Adult (18-25)", "Adult (25+)"],
    index=0
)

# ===============================
# USER INPUT
# ===============================

user_input = st.text_input(
    "Describe the kind of book you want (example: a girl who has to fight, magical adventure):"
)

# ===============================
# KEYWORD EXTRACTION
# ===============================

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
    'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'as', 'if', 'because', 'until', 'while', 'like', 'want',
    'wants', 'kind', 'type', 'book', 'books', 'story', 'stories', 'read',
    'something', 'anything', 'everything', 'nothing', 'someone', 'anyone'
}

def extract_keywords(text):
    """Extract meaningful keywords from conversational text."""
    import re
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(keywords) if keywords else text

# ===============================
# GOOGLE BOOKS API FUNCTION
# ===============================

def fetch_books_googlebooks(query, max_results=40):
    books = []
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "key": GOOGLE_BOOKS_API
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching data from Google Books API (status {response.status_code})")
        try:
            st.json(response.json())
        except:
            pass
        return books

    data = response.json()

    for item in data.get("items", []):
        volume_info = item.get("volumeInfo", {})
        description = volume_info.get("description")

        if not description:
            continue

        title = volume_info.get("title", "Unknown Title")
        authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
        cover_id = volume_info.get("imageLinks", {}).get("thumbnail", None)
        published_date = volume_info.get("publishedDate", "N/A")

        books.append({
            "title": title,
            "author": authors,
            "description": description,
            "cover_id": cover_id,
            "published_date": published_date
        })

    return books

# ===============================
# GEMINI AI RANKING FUNCTION
# ===============================

def rank_books_with_gemini(user_query, books, age_range):
    """Use Gemini to intelligently rank books based on user preferences."""

    books_text = ""
    for i, book in enumerate(books[:10]):  # Limit to 10 books for API efficiency
        books_text += f"\n{i+1}. Title: {book['title']}\n"
        books_text += f"   Author: {book['author']}\n"
        books_text += f"   Description: {book['description'][:500]}...\n"

    age_context = f"The reader's age range is: {age_range}" if age_range else "No specific age range specified."

    prompt = f"""You are a book recommendation expert. A user is looking for a book with this description:

"{user_query}"

{age_context}

Here are the available books:
{books_text}

Rank the TOP 3 books that best match the user's request. For each book, provide:
1. The book number (from the list above)
2. A match score from 0.0 to 1.0
3. A brief explanation of why this book matches

Format your response EXACTLY like this (just the numbers and scores, one per line):
BOOK_NUMBER|SCORE|REASON
BOOK_NUMBER|SCORE|REASON
BOOK_NUMBER|SCORE|REASON

Example:
3|0.85|Features a strong female protagonist in a magical world
7|0.72|Adventure story with fantasy elements
1|0.65|Coming-of-age tale with action sequences"""

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        result_text = response.text.strip()

        ranked_books = []
        for line in result_text.split('\n'):
            if '|' in line:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    try:
                        book_idx = int(parts[0].strip()) - 1
                        score = float(parts[1].strip())
                        reason = parts[2].strip() if len(parts) > 2 else ""
                        if 0 <= book_idx < len(books):
                            book = books[book_idx].copy()
                            book['gemini_reason'] = reason
                            ranked_books.append((score, book))
                    except (ValueError, IndexError):
                        continue

        return ranked_books[:3] if ranked_books else None
    except Exception as e:
        st.warning(f"Gemini ranking failed: {e}")
        return None

# ===============================
# TF-IDF RANKING FUNCTION
# ===============================

def rank_books_with_tfidf(user_query, books):
    """Use TF-IDF to rank books based on text similarity."""
    corpus = [user_query] + [f"{book['title']} {book['description']}" for book in books]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(vectors[0], vectors[1:])[0]
    ranked_books = sorted(zip(similarities, books), reverse=True, key=lambda x: x[0])
    return ranked_books[:3]

# ===============================
# MAIN LOGIC
# ===============================

if user_input:
    search_query = extract_keywords(user_input)

    with st.spinner("Searching for books..."):
        books = fetch_books_googlebooks(search_query)

    if not books:
        st.warning("No books found. Try a different description.")
    else:
        if ai_method == "Gemini AI (Smarter)" and gemini_available:
            with st.spinner("Gemini AI is analyzing books..."):
                ranked_books = rank_books_with_gemini(user_input, books, age_range)

            if not ranked_books:
                st.info("Falling back to TF-IDF method...")
                ranked_books = rank_books_with_tfidf(user_input, books)
                ai_label = "TF-IDF"
            else:
                ai_label = "Gemini AI"
        else:
            ranked_books = rank_books_with_tfidf(user_input, books)
            ai_label = "TF-IDF"

        st.subheader(f"AI Recommended Books ({ai_label})")

        for score, book in ranked_books:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(f"**Published Date:** {book['published_date']}")
            st.write(f"**Age Range:** {age_range if age_range else 'N/A'}")
            st.write(book["description"])
            st.write(f"AI Match Score: {round(score, 2)}")

            if 'gemini_reason' in book and book['gemini_reason']:
                st.write(f"*Why this book:* {book['gemini_reason']}")

            if book["cover_id"]:
                st.image(book["cover_id"], width=150)
            st.write("---")

# ===============================
# RESET BUTTON
# ===============================

if st.button("Start Again"):
    st.rerun()
