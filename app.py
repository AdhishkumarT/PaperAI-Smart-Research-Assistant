import streamlit as st
from sentence_transformers import SentenceTransformer
import PyPDF2
import numpy as np
import faiss
import os

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = faiss.IndexFlatL2(384)  # 384 is the embedding size for 'all-MiniLM-L6-v2'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to get embeddings for the document text
def get_document_embeddings(text):
    sentences = text.split("\n")  # Split text into sentences or paragraphs
    embeddings = model.encode(sentences)  # Get sentence embeddings
    return sentences, embeddings

# Streamlit file uploader
st.title("AI Research Paper Query System")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question related to the paper:")

if uploaded_file is not None and query:
    # Step 1: Extract text from the uploaded PDF
    document_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Generate document embeddings
    sentences, embeddings = get_document_embeddings(document_text)

    # Step 3: Add document embeddings to FAISS index
    embeddings = np.array(embeddings).astype('float32')  # Ensure it's float32
    index.add(embeddings)  # Add to the FAISS index

    # Step 4: Process the query
    query_embedding = model.encode([query]).astype('float32')  # Get query embedding

    # Step 5: Search for the most similar document section using FAISS
    D, I = index.search(query_embedding, k=3)  # Get top 3 closest sentences/paragraphs

    # Display the most relevant document sections
    st.write("Relevant sections from the paper:")
    for i in I[0]:
        st.write(f"- {sentences[i]}")

    # Step 6: Generate an answer based on the retrieved text (optional, using a model like GPT-3 or similar)
    # You can use OpenAI's API for a more detailed answer generation, or you could use simple text processing
    generated_answer = "This would be the answer generated based on the relevant sections."

    # Step 7: Display the generated answer
    st.write("Generated Answer:", generated_answer)

else:
    st.write("Please upload a PDF file and enter a query.")
