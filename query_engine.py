import os
from groq import Groq
from dotenv import load_dotenv
from indexer import search_frames

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_query(user_query: str, n_results: int = 5) -> dict:
    """
    Takes a natural language query, searches the vector store,
    and uses Groq LLM to generate a detailed answer with timestamps.
    """
    print(f"\nSearching for: '{user_query}'")

    # Step 1: Search ChromaDB for relevant frames
    relevant_frames = search_frames(user_query, n_results=n_results)

    if not relevant_frames:
        return {
            "query": user_query,
            "answer": "No video has been indexed yet. Please upload and process a video first.",
            "relevant_frames": []
        }

    # Step 2: Build context from retrieved frames
    context_parts = []
    for frame in relevant_frames:
        context_parts.append(
            f"[Timestamp {frame['timestamp']}] "
            f"(Relevance: {frame['relevance_score']:.0%})\n"
            f"{frame['description']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Ask Groq LLM to answer based on retrieved context
    system_prompt = """You are a workplace safety AI assistant analyzing CCTV footage.
You are given descriptions of video frames with timestamps.
Answer the user's question based ONLY on the frame descriptions provided.
Always mention specific timestamps when referring to events.
If you mention an incident, always include when it happened (timestamp).
Be concise but thorough."""

    user_message = f"""Based on these video frame descriptions, answer the question:

QUESTION: {user_query}

RELEVANT FRAMES FROM VIDEO:
{context}

Provide a clear answer referencing specific timestamps."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return {
        "query": user_query,
        "answer": answer,
        "relevant_frames": relevant_frames
    }