import streamlit as st
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from textblob import TextBlob  # For spell correction
import pickle

# MongoDB client setup
client = MongoClient("mongodb+srv://Haran:NGjWcKG55e3K83SB@cluster0.0ttlk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["searchEngine"]
collection = db["Recipe"]
pdf_collection = db['pdf_files']  # Collection for PDFs

# Load pre-trained model for generating query embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load TF-IDF scores for query expansion
with open('./idf_values.pkl', 'rb') as file:
    tfidf_scores = pickle.load(file)

with open('./word2vec_model.pkl', 'rb') as file:
    expanded_model = pickle.load(file)


def check_spelling(query):
    # Create a TextBlob object
    blob = TextBlob(query)
    
    # Get the corrected text
    corrected_text = blob.correct()
    
    # Check if original and corrected texts are the same
    if query == str(corrected_text):
        return query
    else:
        return f"Showing results for: {corrected_text}", str(corrected_text)

# Cosine similarity search
def search_with_cosine_similarity(query_embedding, top_n=10):
    docs = list(collection.find({}, {"RecipeId": 1, "Name": 1, "encoded_vector": 1}))
    if not docs:
        return []

    doc_embeddings = np.array([doc['encoded_vector'] for doc in docs])
    doc_ids = [doc['RecipeId'] for doc in docs]
    doc_names = [doc['Name'] for doc in docs]

    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_n_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in top_n_indices:
        results.append({
            'RecipeId': doc_ids[idx],
            'Name': doc_names[idx],
            'Similarity': similarities[idx]
        })
    return results

# Nearest Neighbors search
def search_with_nearest_neighbors(query_embedding, top_n=10):
    docs = list(collection.find({}, {"RecipeId": 1, "Name": 1, "encoded_vector": 1}))
    if not docs:
        return []

    doc_embeddings = np.array([doc['encoded_vector'] for doc in docs])
    doc_ids = [doc['RecipeId'] for doc in docs]
    doc_names = [doc['Name'] for doc in docs]

    neigh = NearestNeighbors(n_neighbors=top_n, metric='cosine')
    neigh.fit(doc_embeddings)
    distances, indices = neigh.kneighbors([query_embedding], n_neighbors=top_n)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            'RecipeId': doc_ids[idx],
            'Name': doc_names[idx],
            'Similarity': 1 - distance
        })
    return results

# Search using clusters
def search_using_clusters(query_embedding, top_n=10):
    documents = list(collection.find({}, {"encoded_vector": 1, "cluster": 1}))
    doc_embeddings = np.array([doc['encoded_vector'] for doc in documents])
    clusters = np.array([int(doc['cluster']) for doc in documents])

    cluster_distances = cosine_similarity([query_embedding], doc_embeddings)[0]
    closest_cluster = clusters[np.argmax(cluster_distances)]

    filtered_docs = list(collection.find({"cluster": int(closest_cluster)}, {"RecipeId": 1, "Name": 1, "encoded_vector": 1}))

    doc_embeddings = np.array([doc['encoded_vector'] for doc in filtered_docs])
    doc_ids = [doc['RecipeId'] for doc in filtered_docs]
    doc_names = [doc['Name'] for doc in filtered_docs]

    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_n_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in top_n_indices:
        results.append({
            'RecipeId': doc_ids[idx],
            'Name': doc_names[idx],
            'Similarity': similarities[idx]
        })
    return results

# Search PDFs using cosine similarity
def search_pdf(query_embedding, top_n=10):
    pdf_data = list(pdf_collection.find({}))
    pdf_embeddings = [np.array(pdf['embedding']) for pdf in pdf_data]

    similarities = cosine_similarity([query_embedding], pdf_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    top_pdfs = []
    for index in top_indices:
        top_pdfs.append({
            'name': pdf_data[index]['file_name'],
            'download_link': pdf_data[index]['download_link'],
            'Similarity': similarities[index]
        })

    return top_pdfs

# Streamlit UI
st.title("Recipe and PDF Search Engine")

# User input for search query
query = st.text_input("Enter your search query:", placeholder="e.g., Italian pasta, healthy recipes...")

spelling_result = check_spelling(query)



search_type = st.selectbox("Select Search Type:", ["Cosine Similarity", "Nearest Neighbors", "Cluster-Based", "PDF Search"])

# If spelling_result is a tuple, it indicates that a correction was made
if isinstance(spelling_result, tuple):
    st.subheader(spelling_result[0])  # Display "Showing results for: ..."
    correct_query = spelling_result[1]
else:
    correct_query = spelling_result


if correct_query:
    query_embedding = model.encode([correct_query])[0]

    if search_type == "Cosine Similarity":
        results = search_with_cosine_similarity(query_embedding)
        st.subheader("Cosine Similarity Search Results:")
        for result in results:
            st.write(f"{result['Name']} (Similarity: {result['Similarity']:.2f})")

    elif search_type == "Nearest Neighbors":
        results = search_with_nearest_neighbors(query_embedding)
        st.subheader("Nearest Neighbors Search Results:")
        for result in results:
            st.write(f"{result['Name']} (Similarity: {result['Similarity']:.2f})")

    elif search_type == "Cluster-Based":
        results = search_using_clusters(query_embedding)
        st.subheader("Cluster-Based Search Results:")
        for result in results:
            st.write(f"{result['Name']} (Similarity: {result['Similarity']:.2f})")

    elif search_type == "PDF Search":
        pdf_results = search_pdf(query_embedding)
        st.subheader("PDF Search Results:")
        for pdf in pdf_results:
            st.write(f"[{pdf['name']}]({pdf['download_link']}) (Similarity: {pdf['Similarity']:.2f})")

# Footer or additional information
st.write("---")
