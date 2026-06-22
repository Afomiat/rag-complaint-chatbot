def get_prompt_template(question: str, context: str) -> str:
    """
    Builds the prompt that gets sent to the LLM.
    
    Args:
        question: The user's plain-English question
        context: The retrieved complaint chunks joined together
    
    Returns:
        A formatted prompt string
    """
    return f"""You are a financial analyst assistant for CrediTrust Financial. \
Your job is to answer questions about customer complaints.

Use ONLY the complaint excerpts provided below to formulate your answer. 
Do not use any outside knowledge. If the context does not contain enough 
information to answer the question, say: "I don't have enough information 
in the available complaints to answer this question."

Retrieved Complaint Excerpts:
{context}

Question: {question}

Answer:"""