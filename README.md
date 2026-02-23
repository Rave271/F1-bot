# 🏎️ F1 Constructor Championship Predictor

A Python application that predicts future F1 constructor championship standings using **Llama 3 8B** via Ollama with **RAG (Retrieval Augmented Generation)**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green)

---

## 📁 Project Structure

```
F1 bot/
├── data/
│   ├── constructor_standings.csv   # Historical constructor standings
│   ├── constructors.csv            # Constructor names and details
│   └── races.csv                   # Race information (year, round)
├── vector_db/                      # ChromaDB storage (auto-generated)
├── data_cleaning.py                # Data loading & cleaning
├── vector_store.py                 # Vector database management
├── predict.py                      # LLM prediction with RAG
├── app.py                          # Streamlit web UI
└── README.md
```

---

## 🛠️ Requirements

### Python Packages
```bash
pip install pandas requests streamlit chromadb sentence-transformers
```

### External
- **Ollama** with Llama 3 8B model installed
  ```bash
  ollama pull llama3:8b
  ```

---

## 🚀 How to Run

### Step 1: Build Vector Database (First time only)
```bash
cd "F1 bot"
python3 vector_store.py
```
This creates embeddings for all F1 historical data in ChromaDB (~998 documents).

### Step 2: Start Ollama
```bash
ollama serve
```

### Step 3: Run the Application

**Option A: Web UI (Recommended)**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

**Option B: Command Line**
```bash
python3 predict.py
```

---

## 🏗️ Architecture

This project uses **RAG (Retrieval Augmented Generation)**:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   F1 CSV Data   │────▶│  Sentence        │────▶│   ChromaDB      │
│                 │     │  Transformers    │     │   Vector Store  │
└─────────────────┘     │  (Embeddings)    │     └────────┬────────┘
                        └──────────────────┘              │
                                                          │ Semantic Search
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Prediction    │◀────│   Llama 3 8B     │◀────│   Retrieved     │
│   Output        │     │   (via Ollama)   │     │   Context       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. Historical F1 data is embedded using `all-MiniLM-L6-v2`
2. Embeddings stored in ChromaDB vector database
3. Semantic search retrieves relevant historical context
4. Retrieved context + prompt sent to Llama 3 8B
5. LLM generates predictions based on relevant data

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **RAG Predictions** | Semantic search for relevant historical context |
| 🗄️ **Vector Database** | ChromaDB for efficient embedding storage |
| 🔄 **Mode Toggle** | Switch between RAG and legacy mode |
| 🔍 **Custom Query** | Focus predictions on specific aspects (e.g., "Red Bull dominance") |
| 📊 **Data Visualization** | View historical standings in the UI |
| 🎛️ **Temperature Control** | Adjust model creativity |
| 🎨 **F1-Themed UI** | Clean Streamlit interface |

---

## 🔍 Custom Query Examples

The custom query parameter focuses RAG retrieval on specific aspects:

| Query | Effect |
|-------|--------|
| `Red Bull dominance` | More context about Red Bull's championships |
| `Ferrari wins` | Focus on Ferrari's historical victories |
| `midfield battle` | Data about teams in positions 4-7 |
| `Mercedes decline` | Context about Mercedes losing ground |

---

## 📜 File Execution Order

1. **`vector_store.py`** - Build vector database (run once)
2. **`data_cleaning.py`** - Loads/cleans CSV data (imported automatically)
3. **`predict.py`** - Makes predictions using RAG + Ollama
4. **`app.py`** - Runs the Streamlit web interface

---

## 📄 License

MIT License - Feel free to use and modify!
