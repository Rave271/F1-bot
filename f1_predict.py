import pandas as pd
import requests
import json

# Load data
constructor_standings = pd.read_csv('data/constructor_standings.csv')
races = pd.read_csv('data/races.csv')
constructors = pd.read_csv('data/constructors.csv')

# Merge to get year and round
merged = pd.merge(constructor_standings, races[['raceId', 'year', 'round']], on='raceId')

# Clean: drop unnecessary columns, handle missing values
cleaned = merged[['year', 'round', 'constructorId', 'points', 'position', 'wins']].copy()
cleaned = cleaned.dropna()  # Assuming no missing, but just in case

# Aggregate per season per constructor: take the last round's data
seasonal = cleaned.loc[cleaned.groupby(['year', 'constructorId'])['round'].idxmax()]

# Get constructor names
seasonal = pd.merge(seasonal, constructors[['constructorId', 'name']], on='constructorId')

# Sort by year and points descending
seasonal = seasonal.sort_values(['year', 'points'], ascending=[True, False])

# Filter to recent years, say 2020-2023
seasonal = seasonal[seasonal['year'] >= 2020]

# Prepare data for LLM
historical_data = ""
for year in range(2020, 2024):
    year_data = seasonal[seasonal['year'] == year]
    historical_data += f"Season {year}:\n"
    for _, row in year_data.iterrows():
        historical_data += f"{row['name']}: {row['points']} points, position {row['position']}, {row['wins']} wins\n"
    historical_data += "\n"

# Prompt for prediction
prompt = f"Based on the following recent Formula 1 constructor standings, predict the 2024 season standings in the same format:\n\n{historical_data}\n2024 prediction:"

# Call Llama 3 8B via Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8
        }
    }
)

result = response.json()
prediction = result.get("response", "")

print("Prediction for 2024:")
print(prediction)