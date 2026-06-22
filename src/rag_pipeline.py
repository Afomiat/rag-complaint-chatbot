import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root before anything else
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

from src.retriever import ComplaintRetriever
from src.generator import LLMGenerator
from src.prompt import get_prompt_template
from typing import Dict


class RAGPipeline:
    """
    The complete RAG pipeline.
    Combines the retriever and generator into one clean interface.

    Usage:
        pipeline = RAGPipeline(vector_store_path='./vector_store/chroma_db')
        result = pipeline.ask("Why are people unhappy with credit cards?")
        print(result['answer'])
    """

    def __init__(self, vector_store_path: str, top_k: int = 5):
        """
        Args:
            vector_store_path: Path to ChromaDB vector store
            top_k: Number of complaint chunks to retrieve per query
        """
        print("Initializing RAG Pipeline...")
        self.retriever = ComplaintRetriever(vector_store_path)
        self.generator = LLMGenerator()
        self.top_k = top_k
        print("RAG Pipeline ready!")

    def ask(self, question: str) -> Dict:
        """
        The main method — takes a question and returns
        a grounded answer with sources.

        Args:
            question: Plain-English question from the user

        Returns:
            Dict with keys:
                'answer'  — the LLM generated answer
                'sources' — list of retrieved chunks with metadata
                'context' — the full context string sent to LLM
        """
        print(f"\nQuestion: {question}")

        # Step 1 — retrieve relevant complaint chunks
        print("Retrieving relevant complaints...")
        retrieved_chunks = self.retriever.retrieve(
            question=question,
            top_k=self.top_k
        )

        # Step 2 — format chunks into context string
        context = self.retriever.format_context(retrieved_chunks)

        # Step 3 — build the prompt
        prompt = get_prompt_template(
            question=question,
            context=context
        )

        # Step 4 — send to LLM and get answer
        print("Generating answer...")
        answer = self.generator.generate(prompt)

        # Step 5 — return everything
        return {
            'answer': answer,
            'sources': retrieved_chunks,
            'context': context
        }