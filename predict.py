import requests
from data_cleaning import get_seasonal_data, format_historical_data

def predict_season(historical_data, target_year=2024, model="llama3:8b", temperature=0.8):
    """Use Llama 3 via Ollama to predict constructor standings."""
    prompt = f"""Based on the following F1 constructor standings, predict the {target_year} season.

{historical_data}

Give ONLY:
1. A ranked list of teams with predicted points (1 line per team)
2. A 2-sentence summary

{target_year} Predicted Standings:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be loading or too slow."
    except Exception as e:
        return f"Error: {str(e)}"

def predict_with_rag(query, target_year=2024, model="llama3:8b", temperature=0.8):
    """Use RAG to get context and predict with Llama 3."""
    from vector_store import get_context_for_prediction, query_similar
    
    # Get relevant context from vector store
    context = get_context_for_prediction(target_year)
    
    # Also query for specific patterns if user has a query
    if query:
        specific_results = query_similar(query, n_results=5)
        extra_context = "\n".join(specific_results['documents'][0])
        context = f"{context}\n\nAdditional relevant info:\n{extra_context}"
    
    prompt = f"""You are an F1 expert. Using the following historical data retrieved from our database, predict the {target_year} F1 Constructor Championship.

RETRIEVED CONTEXT:
{context}

Based on this data, predict the {target_year} season standings.

Give ONLY:
1. A ranked list of all 10 teams with predicted points (1 line per team)
2. A 2-sentence summary explaining your prediction

{target_year} Predicted Standings:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 400
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received"), context
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve).", ""
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be loading or too slow.", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

def get_prediction(start_year=2020, end_year=2023, target_year=2024, temperature=0.8):
    """Get F1 constructor championship prediction (legacy method)."""
    seasonal = get_seasonal_data(start_year, end_year)
    historical_data = format_historical_data(seasonal, start_year, end_year)
    prediction = predict_season(historical_data, target_year, temperature=temperature)
    return prediction, historical_data

def get_rag_prediction(query="", target_year=2024, temperature=0.8):
    """Get F1 constructor championship prediction using RAG."""
    prediction, context = predict_with_rag(query, target_year, temperature=temperature)
    return prediction, context

if __name__ == "__main__":
    print("=" * 50)
    print("Testing RAG-based prediction...")
    print("=" * 50)
    
    prediction, context = get_rag_prediction(target_year=2024)
    
    print("\nRetrieved Context (from vector DB):")
    print("-" * 40)
    print(context[:1000] + "..." if len(context) > 1000 else context)
    
    print("\n" + "=" * 50)
    print("Prediction for 2024:")
    print("=" * 50)
    print(prediction)
    print("\nPrediction for 2024:")
    print(prediction)
