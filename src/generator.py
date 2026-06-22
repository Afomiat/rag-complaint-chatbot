import os
import time
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')


class LLMGenerator:
    """
    Handles sending the prompt to Groq (Llama 3) and getting a response.
    Groq is free, fast, and uses simple HTTP.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Args:
            model_name: Groq model to use
                       Options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
        """
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure it is in your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        print(f"LLM ready: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Sends prompt to Groq and returns generated answer.
        Automatically retries if rate limited.

        Args:
            prompt: Full prompt string (question + context)
            max_tokens: Maximum length of generated answer

        Returns:
            LLM generated answer as string
        """
        max_retries = 3
        wait_seconds = 30

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial analyst assistant for CrediTrust Financial. Answer questions about customer complaints using only the provided context."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    print(f"Rate limited. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                return f"Error generating response: {error_msg}"