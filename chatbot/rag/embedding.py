# chatbot/rag/embedding.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def embed_texts(texts):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    return embeddings_model.embed_documents(texts)
