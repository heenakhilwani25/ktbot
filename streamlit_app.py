import os
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import httpx
import requests
import json
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import boto3
from langchain.chains.question_answering import load_qa_chain
# Initialize a session using Amazon S3
import botocore
import io
import base64
import streamlit as st

bucket_name = 'emd-forecast'
file_key="gtn/input/cacert.pem"
prefix = 'gtn/input/knowledge_base/'  # Replace with your actual prefix
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    verify=False
)

def load_docx(file_content):
    # Read the DOCX file content and return the text
    doc = Document(io.BytesIO(file_content))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def load_docs_from_s3(bucket_name, prefix):
    documents = []
    try:
        # List all objects in the S3 bucket with the specified prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.docx'):
                # Read the file content from S3
                file_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                file_content = file_obj['Body'].read()
                text = load_docx(file_content)
                if text:
                    documents.append((obj['Key'], text))  # Store filename and text together
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
    return documents

# Split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for doc_id, doc in documents:
        splits = text_splitter.split_text(doc)
        docs.extend([(doc_id, split) for split in splits])
    return docs

# Load documents from a directory
documents = load_docs_from_s3(bucket_name, prefix)
print(f"Number of documents loaded: {len([d[0] for d in documents])}")

# Split documents into chunks
docs = split_docs(documents)
print(f"Number of split documents: {len(docs)}")
embeddings = []
openai.api_key = st.secrets["api_key"]

for doc_id, doc in docs:
    text = doc
    embedding_response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    print(embedding_response)
    if embedding_response is not None:
        embedding = embedding_response.data[0].embedding
        embeddings.append((embedding, doc_id, text))

# Initialize HTTP client
p_api_key = st.secrets["p_api_key"]

base_url = st.secrets["base_url"]


upsert_endpoint = f"{base_url}/vectors/upsert"
query_endpoint = f"{base_url}/vectors/query"

# Headers with API key
headers = {
    "Content-Type": "application/json",
    "Api-Key": p_api_key
}

dimension = 1536
for embed, doc_id, text in embeddings:
    assert len(embed) == dimension, "Embedding dimension mismatch"

# Prepare the data to be added to the index
data = {
    "vectors": [
        {
           "id": f"{doc_id}-{i}",
            "values": embed,
            "metadata": {"text": text} 
        }
        for i, (embed, doc_id, text) in enumerate(embeddings)
    ]
}

# Make the POST request to upsert data
response = requests.post(upsert_endpoint, headers=headers, json=data, verify=False)

# Check if the request was successful
if response.status_code == 200:
    print("Data upserted successfully.")
else:
    print(f"Failed to upsert data: {response.status_code} - {response.reason}")
    print("Response content:", response.text)
    
    
doc_text_dict = {f"{doc_id}-{i}": text for i, (embed, doc_id, text) in enumerate(embeddings)}
# Preprocess the id to extract a consistent format
# Preprocess the id to extract a consistent format
def preprocess_id(id_str):
    if id_str.startswith("page_content="):
        return id_str.split("=")[1].strip("'")
    else:
        return id_str



# Update the get_similar_docs function to preprocess the id before retrieval
def get_similar_docs(query, k=2, score=False):
    # Generate the embedding for the query
    query_embedding_response = client.embeddings.create(input=query, model="text-embedding-ada-002-v2")
    query_embedding = query_embedding_response.data[0].embedding
    
    # Search the Pinecone index for similar documents
    query_payload = {
        "top_k": k,
        "include_values": score,
        "vector": query_embedding
    }
    query_endpoint = f"{base_url}/query"
    query_response = requests.post(query_endpoint, headers=headers, json=query_payload, verify=False)
    
    # Extract and return the similar documents
    if query_response.status_code == 200:
        search_results = query_response.json()
        print(search_results)
        similar_docs = [doc_text_dict[preprocess_id(match['id'])] for match in search_results['matches']]
        return similar_docs
    else:
        print(f"Failed to retrieve similar documents: {query_response.status_code} - {query_response.reason}")
        print("Response content:", query_response.text)
        return []


def get_answer(query):
    similar_docs = get_similar_docs(query)
    combined_message = f"Question: {query}\nDocuments: {similar_docs}"
    client = OpenAI(
    # This is the default and can be omitted
    api_key =  st.secrets["api_key"]
    )

    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": combined_message
                }
            ]
    )

    ChatGPT_reply = chat_completion.choices[0].message.content
    return ChatGPT_reply





file_key = 'gtn/input/logo_ktbot.PNG'

# Download the file from S3
response = s3.get_object(Bucket=bucket_name, Key=file_key)
file_content = response['Body'].read()

# Encode the file content in base64
png_base64 = base64.b64encode(file_content).decode('utf-8')
png_data_url = f"data:image/png;base64,{png_base64}"


# Streamlit App
st.set_page_config(page_title="KT Bot", layout="wide")

st.sidebar.image(png_data_url, use_column_width=True)

st.title("KT Bot")
st.write("Ask any question related to the knowledge base")

user_input = st.text_input("Query:", "")
if st.button("SEND"):
    if user_input:
        st.write("Processing...")
        try:
            response = get_answer(user_input)
            st.write("**Answer:**")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
