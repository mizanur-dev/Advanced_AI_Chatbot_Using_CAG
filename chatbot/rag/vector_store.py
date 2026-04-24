# chatbot/rag/vector_store.py

from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def clear_namespace(namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"Error clearing namespace: {e}")


def upsert_chunks(chunks, embeddings, namespace):
    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{namespace}_{i}",
            "values": emb,
            "metadata": {
                "text": chunk
            }
        })

    index.upsert(vectors=vectors, namespace=namespace)