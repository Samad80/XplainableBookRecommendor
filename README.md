
# 📚 Explainable Book Recommender

This Streamlit app is an interactive, explainable book recommender system. It suggests books based on title and author similarity using TF-IDF vectorization and cosine similarity. It also leverages SHAP to explain why a particular book was recommended.

## 🔧 Features

* **Book Recommendations**: Based on TF-IDF and cosine similarity of book titles and authors.
* **Explainability**: Uses SHAP to explain what features (words) influenced a recommendation.
* **Interactive UI**: Built with Streamlit for easy interaction.

## 🧠 How It Works

1. **Data Loading**: The app loads a CSV file named `books.csv` containing `title` and `authors` columns.
2. **TF-IDF Vectorization**: Converts combined `title` and `authors` text into numerical features (limited to 1000 features).
3. **Cosine Similarity**: Calculates similarity between books.
4. **Recommendation**: Suggests top 5 similar books based on the selected title.
5. **SHAP Explainability**: Explains recommendations using a Ridge regression model trained on the TF-IDF features.

## 🗂️ File Structure

* `str.py`: Main app file to be run with Streamlit.
* `books.csv`: Required dataset with at least `title` and `authors` columns.

## ▶️ Getting Started

### 1. Install Dependencies

```bash
pip install streamlit pandas scikit-learn shap matplotlib
```

### 2. Add Dataset

Ensure a `books.csv` file is present in the same directory. It should include:

```csv
title,authors
Book Title 1,Author Name
...
```

### 3. Run the App

```bash
streamlit run str.py
```

## 📈 Example

* Select a book from the dropdown.
* Click the "Recommend and Explain" button.
* View top 5 similar book recommendations.
* See a SHAP bar chart explaining the recommendations.

## ✅ Requirements

* Python 3.7+
* Streamlit
* Pandas
* scikit-learn
* SHAP
* Matplotlib

## 📌 Notes

* The SHAP explanation is generated using a Ridge regression approximation on similarity scores.
* The vectorizer is limited to 1000 features to maintain interpretability.

