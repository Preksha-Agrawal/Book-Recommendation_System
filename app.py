from flask import Flask, render_template, request
from urllib.parse import quote
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.jinja_env.filters['urlencode'] = lambda u: quote(str(u))

df           = joblib.load("model/books.pkl")
tfidfnum     = joblib.load("model/tfidf_matrix.pkl")
genre_matrix = joblib.load("model/genre_matrix.pkl")

if not sp.issparse(tfidfnum):
    tfidfnum = sp.csr_matrix(tfidfnum)
if not sp.issparse(genre_matrix):
    genre_matrix = sp.csr_matrix(genre_matrix)

IGNORE_GENRES = {
    "Childrens", "Middle Grade",
    "Audiobook", "Nonfiction", "Adult", "Contemporary"
}

IMPORTANT_GENRES = {
    "Fantasy", "Magic", "Adventure", "Historical Fiction",
    "Science Fiction", "Horror", "Mystery", "Romance",
    "Thriller", "Classics", "War", "Fiction", "Historical",
    "Gothic", "Dystopia", "Literature", "Humor"
}


def rec(bookTitle):

    matches = df[df['Book'].str.contains(bookTitle, case=False, na=False)]

    if matches.empty:
        return ["Book not found"]

    pos          = df.index.get_loc(matches.index[0])
    genres       = set(matches.iloc[0]["Genres_list"])
    title        = matches.iloc[0]["Book"]
    query_author = matches.iloc[0]["Author"]
    series       = ""

    if "(" in title:
        series = title.split("(")[1].split(",")[0].strip().lower()

    tfidf_sim = cosine_similarity(tfidfnum[pos], tfidfnum).flatten()
    genre_sim  = cosine_similarity(genre_matrix[pos], genre_matrix).flatten()
    combined   = 0.7 * tfidf_sim + 0.3 * genre_sim

    sim = sorted(list(enumerate(combined)), key=lambda x: x[1], reverse=True)

    genres_filtered = {g for g in genres if g not in IGNORE_GENRES}
    if not genres_filtered:
        genres_filtered = genres

    recs        = []
    seen_titles = set()
    seen_series = set()

    for i in sim[1:]:

        bookIndex  = i[0]
        score      = i[1]
        name       = df.iloc[bookIndex]['Book']
        name_lower = name.lower()
        author     = df.iloc[bookIndex]['Author']

        
        if bookTitle.lower() in name_lower:
            continue

        
        if series and series in name_lower:
            continue

        if name in seen_titles:
            continue

        candidate_series = ""
        if "(" in name:
            candidate_series = name.split("(")[1].split(",")[0].strip().lower()
        if candidate_series and candidate_series in seen_series:
            continue

        rating     = df.iloc[bookIndex]['avgr']
        popular    = df.iloc[bookIndex]['popular']
        num_rating = df.iloc[bookIndex]['numr']

        genre          = set(df.iloc[bookIndex]['Genres_list'])
        genre_filtered = {g for g in genre if g not in IGNORE_GENRES}

        if not genre_filtered:
            continue

        comGenre = genres_filtered.intersection(genre_filtered)

        if not comGenre:
            continue
        if num_rating < 50000:
            continue

        wt = len(comGenre)
        for g in comGenre:
            if g in IMPORTANT_GENRES:
                wt += 2

        final = score * (rating / 5) * (1 + wt / 5) * (1 + popular / 10)

        recs.append({
            "title":  name,
            "author": author,
            "rating": float(rating),
            "votes":  int(num_rating),
            "genres": ", ".join(comGenre),
            "score":  round(float(final), 4)
        })

        seen_titles.add(name)
        if candidate_series:
            seen_series.add(candidate_series)

        if len(recs) == 500:
            break

    recs = sorted(recs, key=lambda x: (x["score"] * 0.5 + x["rating"] / 5 * 0.5), reverse=True)[:5]

    if not recs:
        return ["No recommendations found for this book."]

    return recs


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    searched_book   = ""

    if request.method == "POST":
        book            = request.form["book"]
        searched_book   = book
        recommendations = rec(book)

    return render_template("index.html", recs=recommendations, searched_book=searched_book)


if __name__ == "__main__":
    app.run(debug=True)