F1 Constructor Championship Predictor
======================================

A Python application that uses historical F1 constructor standings data to predict
future season results using Llama 3 8B via Ollama with RAG (Retrieval Augmented Generation).


PROJECT STRUCTURE
-----------------
data/
    constructor_standings.csv  - Historical constructor standings by race
    constructors.csv           - Constructor names and details
    races.csv                  - Race information including year and round

vector_db/                     - ChromaDB persistent storage (auto-generated)

data_cleaning.py    - Data loading and cleaning functions
vector_store.py     - Vector database management (ChromaDB + embeddings)
predict.py          - LLM prediction logic with RAG support
app.py              - Streamlit web UI
f1_predict.py       - Original standalone script (deprecated)


REQUIREMENTS
------------
- Python 3.x
- pandas
- requests
- streamlit
- chromadb
- sentence-transformers
- Ollama with Llama 3 8B installed


HOW TO RUN
----------

Step 1: Build Vector Database (First time only)
------------------------------------------------
$ cd "/Users/raghavverma/Downloads/F1 bot"
$ python3 vector_store.py

This creates embeddings for all F1 historical data in ChromaDB.


Step 2: Run the Application
---------------------------

Option A: Web UI (Recommended)
1. Make sure Ollama is running:
   $ ollama serve

2. Run the Streamlit app:
   $ streamlit run app.py

3. Open http://localhost:8501 in your browser


Option B: Command Line
1. Make sure Ollama is running:
   $ ollama serve

2. Run the prediction script:
   $ python3 predict.py


FILE EXECUTION ORDER
--------------------
1. vector_store.py  - Build vector database (run once)
2. data_cleaning.py - Loads and cleans CSV data (imported by other files)
3. predict.py       - Makes predictions using RAG + Ollama
4. app.py           - Runs the Streamlit web interface


ARCHITECTURE
------------
This project uses RAG (Retrieval Augmented Generation):

1. Historical F1 data is embedded using Sentence Transformers (all-MiniLM-L6-v2)
2. Embeddings are stored in ChromaDB vector database
3. When predicting, relevant historical context is retrieved via semantic search
4. Retrieved context is sent to Llama 3 8B along with the prediction prompt
5. LLM generates predictions based on the relevant historical data


FEATURES
--------
- RAG-powered predictions with semantic search
- Vector database for efficient context retrieval
- Toggle between RAG and legacy mode
- Custom query support for focused predictions
- Historical data visualization
- Adjustable temperature for model creativity
- Clean F1-themed UI
