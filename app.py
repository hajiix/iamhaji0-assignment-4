from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all')

# Initialize vectorizer and LSA
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
lsa = TruncatedSVD(n_components=111)

# Create term-document matrix
tdm = vectorizer.fit_transform(newsgroups.data)

# Apply LSA
lsa.fit(tdm)
reduced_tdm = lsa.transform(tdm)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_vector = vectorizer.transform([query])
    query_reduced = lsa.transform(query_vector)
    
    similarities = cosine_similarity(query_reduced, reduced_tdm).flatten()
    top_indices = np.argsort(-similarities)[:5]
    top_similarities = similarities[top_indices]
    top_documents = [newsgroups.data[i] for i in top_indices]
    
    return top_documents, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)