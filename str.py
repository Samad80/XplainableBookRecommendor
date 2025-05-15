# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import linear_kernel
import shap
import matplotlib.pyplot as plt

# --- Load Dataset ---
def load_data():
    books = pd.read_csv("books.csv")
    books = books.dropna(subset=["title", "authors"])
    books["content"] = books["title"] + " " + books["authors"]
    return books

books = load_data()

# --- TF-IDF Vectorizer (limited to 1000 features) ---
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(books['content'])

feature_names = tfidf.get_feature_names_out()
dense_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


# --- Build Cosine Similarity Matrix ---
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(books.index, index=books['title']).drop_duplicates()

# --- Streamlit UI ---
st.title("📚 Explainable Book Recommender")
book_choice = st.selectbox("Select a Book:", books['title'].values)

if st.button("Recommend and Explain"):
    idx = indices[book_choice]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    rec_indices = [i[0] for i in sim_scores]

    st.subheader("Recommended Books:")
    for i in rec_indices:
        st.write(f"**{books.iloc[i]['title']}** by {books.iloc[i]['authors']}")

    # --- SHAP Explainability for the target book ---
    st.subheader("📊 Why is this book recommended?")
    y = cosine_sim[idx]
    model = Ridge()
    model.fit(dense_tfidf, y)

    explainer = shap.LinearExplainer(model, dense_tfidf, feature_perturbation="interventional")
    shap_values = explainer(dense_tfidf.iloc[[idx]])

    # Plot SHAP values
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

