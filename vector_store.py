import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from data_cleaning import load_data, clean_data
import os

# Use sentence-transformers for embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_chroma_client():
    """Get or create ChromaDB client with persistent storage."""
    persist_dir = os.path.join(os.path.dirname(__file__), "vector_db")
    client = chromadb.PersistentClient(path=persist_dir)
    return client

def get_embedding_function():
    """Get the embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

def create_f1_documents():
    """Create documents from F1 data for vector storage."""
    constructor_standings, races, constructors = load_data()
    seasonal = clean_data(constructor_standings, races, constructors)
    
    documents = []
    metadatas = []
    ids = []
    
    # Create documents for each season-constructor combination
    for _, row in seasonal.iterrows():
        year = int(row['year'])
        constructor = row['name']
        points = row['points']
        position = int(row['position'])
        wins = int(row['wins'])
        
        # Create a rich document
        doc = f"In the {year} F1 season, {constructor} finished in position {position} with {points} points and {wins} race wins."
        
        documents.append(doc)
        metadatas.append({
            "year": year,
            "constructor": constructor,
            "points": float(points),
            "position": position,
            "wins": wins
        })
        ids.append(f"{year}_{constructor.replace(' ', '_')}")
    
    # Create season summary documents
    for year in seasonal['year'].unique():
        year_data = seasonal[seasonal['year'] == year].sort_values('points', ascending=False)
        
        champion = year_data.iloc[0]
        top_3 = year_data.head(3)
        
        summary = f"Season {int(year)} Summary: {champion['name']} won the championship with {champion['points']} points and {int(champion['wins'])} wins. "
        summary += f"Top 3: 1. {top_3.iloc[0]['name']}, 2. {top_3.iloc[1]['name']}, 3. {top_3.iloc[2]['name']}."
        
        documents.append(summary)
        metadatas.append({
            "year": int(year),
            "constructor": "SUMMARY",
            "points": 0.0,
            "position": 0,
            "wins": 0
        })
        ids.append(f"{int(year)}_SUMMARY")
    
    return documents, metadatas, ids

def build_vector_store(force_rebuild=False):
    """Build or rebuild the vector store."""
    client = get_chroma_client()
    embedding_fn = get_embedding_function()
    
    # Check if collection exists
    existing_collections = [c.name for c in client.list_collections()]
    
    if "f1_data" in existing_collections and not force_rebuild:
        print("Vector store already exists. Use force_rebuild=True to rebuild.")
        return client.get_collection("f1_data", embedding_function=embedding_fn)
    
    # Delete existing collection if rebuilding
    if "f1_data" in existing_collections:
        client.delete_collection("f1_data")
    
    # Create new collection
    collection = client.create_collection(
        name="f1_data",
        embedding_function=embedding_fn,
        metadata={"description": "F1 Constructor Championship Data"}
    )
    
    # Add documents
    documents, metadatas, ids = create_f1_documents()
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Built vector store with {len(documents)} documents.")
    return collection

def get_collection():
    """Get the F1 data collection."""
    client = get_chroma_client()
    embedding_fn = get_embedding_function()
    return client.get_collection("f1_data", embedding_function=embedding_fn)

def query_similar(query_text, n_results=10, year_filter=None):
    """Query the vector store for similar documents."""
    collection = get_collection()
    
    where_filter = None
    if year_filter:
        where_filter = {"year": {"$gte": year_filter}}
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )
    
    return results

def get_context_for_prediction(target_year, n_recent_years=4):
    """Get relevant context from vector store for prediction."""
    collection = get_collection()
    
    # Query for recent performance data
    query = f"F1 constructor championship standings performance points wins {target_year - 1}"
    
    results = collection.query(
        query_texts=[query],
        n_results=50,
        where={"year": {"$gte": target_year - n_recent_years}}
    )
    
    # Organize results by year
    context_parts = []
    seen = set()
    
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        key = f"{metadata['year']}_{metadata['constructor']}"
        if key not in seen:
            seen.add(key)
            context_parts.append(doc)
    
    return "\n".join(context_parts)

if __name__ == "__main__":
    print("Building vector store...")
    collection = build_vector_store(force_rebuild=True)
    
    print("\nTesting query...")
    results = query_similar("Red Bull championship wins", n_results=5)
    for doc in results['documents'][0]:
        print(f"- {doc}")
