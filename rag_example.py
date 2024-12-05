# ------------------------------- Example 1 ------------------------------------------

# Install required libraries
!pip install kaggle faiss-cpu openai

import os
import zipfile
import openai
import pandas as pd
import numpy as np
import faiss

# Configure Kaggle API client
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'

# Download dataset
!kaggle datasets download -d yelp-dataset/yelp-dataset

# Unzip the downloaded dataset
with zipfile.ZipFile('yelp-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('yelp-dataset')

# Load the dataset
data = pd.read_csv('yelp-dataset/reviews.csv')  # Example path, adjust as needed
documents = data['text'].tolist()  # Assuming the text column contains the documents

# Set OpenAI API key
openai.api_key = 'your_openai_api_key'

# Function to create embeddings for each document
def create_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = openai.Embedding.create(input=doc, model='text-embedding-ada-002')
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

embeddings = create_embeddings(documents)

# Convert embeddings to numpy array
embedding_matrix = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Function to search for similar documents
def search_similar_documents(query, index, documents, top_k=5):
    query_embedding = openai.Embedding.create(input=query, model='text-embedding-ada-002')['data'][0]['embedding']
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    similar_documents = [documents[i] for i in indices[0]]
    
    return similar_documents

# Function to generate response
def generate_response(query, similar_docs, max_tokens=150):
    context = "\n".join(similar_docs)
    prompt = f"Based on the following documents, answer the query: {query}\n\n{context}"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

# Example usage
query = "great food and friendly staff"
similar_docs = search_similar_documents(query, index, documents)
response = generate_response(query, similar_docs)
print(response)

# Generate a limited response
limited_response = generate_response(query, similar_docs, max_tokens=100)
print(limited_response)


# ------------------------------------------------------------------------------------
# ------------------------------- Example 2 ------------------------------------------

# Install required libraries
!pip install kaggle qdrant-client openai

import os
import zipfile
import openai
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Macro definitions
KAGGLE_USERNAME = 'your_kaggle_username'
KAGGLE_KEY = 'your_kaggle_key'
DATASET = 'yelp-dataset/yelp-dataset'
DATASET_PATH = 'yelp-dataset/reviews.csv'
EMBEDDING_MODEL = 'text-embedding-ada-002'
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
OPENAI_API_KEY = 'your_openai_api_key'
MAX_TOKENS = 150
TOP_K = 5

# Configure Kaggle API client
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

# Download dataset
!kaggle datasets download -d {DATASET}

# Unzip the downloaded dataset
with zipfile.ZipFile(f'{DATASET.split("/")[-1]}.zip', 'r') as zip_ref:
    zip_ref.extractall(DATASET.split("/")[0])

# Load the dataset
data = pd.read_csv(DATASET_PATH)
documents = data['text'].tolist()  # Assuming the text column contains the documents

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to create embeddings for each document
def create_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = openai.Embedding.create(input=doc, model=EMBEDDING_MODEL)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

embeddings = create_embeddings(documents)

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Create collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)

# Upload embeddings to Qdrant
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": documents[i]})
    for i in range(len(embeddings))
]
client.upsert(collection_name=COLLECTION_NAME, points=points)

# Function to search for similar documents
def search_similar_documents(query, client, collection_name, top_k=TOP_K):
    query_embedding = openai.Embedding.create(input=query, model=EMBEDDING_MODEL)['data'][0]['embedding']
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    similar_documents = [hit.payload['text'] for hit in search_result]
    return similar_documents

# Function to generate response
def generate_response(query, similar_docs, max_tokens=MAX_TOKENS):
    context = "\n".join(similar_docs)
    prompt = f"Based on the following documents, answer the query: {query}\n\n{context}"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

# Example usage
query = "great food and friendly staff"
similar_docs = search_similar_documents(query, client, COLLECTION_NAME)
response = generate_response(query, similar_docs)
print(response)

# Generate a limited response
limited_response = generate_response(query, similar_docs, max_tokens=100)
print(limited_response)
