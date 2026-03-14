import chromadb
import os

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(collection_name: str = "video_frames"):
    """Get existing collection or create a new one."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def index_frames(enriched_frames: list, collection_name: str = "video_frames"):
    """
    Stores all frame descriptions into ChromaDB vector store.
    Each frame's description becomes a searchable document.
    """
    collection = get_or_create_collection(collection_name)

    # Clear existing data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"Cleared {len(existing['ids'])} old entries from vector store.")

    documents = []
    metadatas = []
    ids = []

    for i, frame in enumerate(enriched_frames):
        doc_id = f"frame_{i}_{frame['frame_number']}"
        documents.append(frame["description"])
        metadatas.append({
            "timestamp": frame["timestamp"],
            "timestamp_seconds": float(frame["timestamp_seconds"]),
            "frame_number": int(frame["frame_number"]),
            "frame_path": frame["frame_path"]
        })
        ids.append(doc_id)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(documents)} frames into ChromaDB!")
    return len(documents)

def search_frames(query: str, n_results: int = 5, collection_name: str = "video_frames"):
    """
    Searches the vector store for frames matching the query.
    Returns top matching frames with their metadata.
    """
    collection = get_or_create_collection(collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    matches = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            matches.append({
                "description": doc,
                "timestamp": results["metadatas"][0][i]["timestamp"],
                "timestamp_seconds": results["metadatas"][0][i]["timestamp_seconds"],
                "frame_number": results["metadatas"][0][i]["frame_number"],
                "frame_path": results["metadatas"][0][i]["frame_path"],
                "relevance_score": 1 - results["distances"][0][i]
            })

    return matches