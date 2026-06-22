import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class ComplaintRetriever:
    """
    Handles semantic search over the ChromaDB vector store.
    Given a plain-English question, finds the most relevant
    complaint chunks by meaning — not keyword matching.
    """

    def __init__(self, vector_store_path: str, collection_name: str = "complaints"):
        """
        Args:
            vector_store_path: Path to the persisted ChromaDB folder
            collection_name: Name of the collection inside ChromaDB
        """
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Connecting to vector store...")
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection(collection_name)

        print(f"Ready! Vector store has {self.collection.count()} chunks.")

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Searches the vector store for the most relevant complaint chunks.

        Args:
            question: The user's plain-English question
            top_k: Number of chunks to retrieve (default 5)

        Returns:
            List of dicts, each with 'text' and 'metadata' keys
        """
        # Step 1 — convert question to vector
        query_vector = self.model.encode(question).tolist()

        # Step 2 — search ChromaDB for closest vectors
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # Step 3 — package results into clean list of dicts
        retrieved = []
        for text, metadata in zip(
            results['documents'][0],
            results['metadatas'][0]
        ):
            retrieved.append({
                'text': text,
                'metadata': metadata
            })

        return retrieved

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Joins retrieved chunks into a single context string
        to be injected into the prompt.

        Args:
            retrieved_chunks: Output from retrieve()

        Returns:
            Formatted string of all chunks with source labels
        """
        context_parts = []

        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk['metadata']
            context_parts.append(
                f"[Complaint {i+1} | "
                f"Product: {meta.get('product_category', 'N/A')} | "
                f"Issue: {meta.get('issue', 'N/A')}]\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(context_parts)