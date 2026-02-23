import pandas as pd
import os

def load_data(data_dir='data'):
    """Load all required CSV files."""
    constructor_standings = pd.read_csv(os.path.join(data_dir, 'constructor_standings.csv'))
    races = pd.read_csv(os.path.join(data_dir, 'races.csv'))
    constructors = pd.read_csv(os.path.join(data_dir, 'constructors.csv'))
    return constructor_standings, races, constructors

def clean_data(constructor_standings, races, constructors):
    """Clean and merge the data."""
    # Merge to get year and round
    merged = pd.merge(constructor_standings, races[['raceId', 'year', 'round']], on='raceId')
    
    # Clean: drop unnecessary columns, handle missing values
    cleaned = merged[['year', 'round', 'constructorId', 'points', 'position', 'wins']].copy()
    cleaned = cleaned.dropna()
    
    # Aggregate per season per constructor: take the last round's data
    seasonal = cleaned.loc[cleaned.groupby(['year', 'constructorId'])['round'].idxmax()]
    
    # Get constructor names
    seasonal = pd.merge(seasonal, constructors[['constructorId', 'name']], on='constructorId')
    
    # Sort by year and points descending
    seasonal = seasonal.sort_values(['year', 'points'], ascending=[True, False])
    
    return seasonal

def get_seasonal_data(start_year=2020, end_year=2023):
    """Get cleaned seasonal data for specified year range."""
    constructor_standings, races, constructors = load_data()
    seasonal = clean_data(constructor_standings, races, constructors)
    seasonal = seasonal[(seasonal['year'] >= start_year) & (seasonal['year'] <= end_year)]
    return seasonal

def format_historical_data(seasonal, start_year=2020, end_year=2023):
    """Format seasonal data as text for LLM."""
    historical_data = ""
    for year in range(start_year, end_year + 1):
        year_data = seasonal[seasonal['year'] == year]
        historical_data += f"Season {year}:\n"
        for _, row in year_data.iterrows():
            historical_data += f"{row['name']}: {row['points']} points, position {row['position']}, {row['wins']} wins\n"
        historical_data += "\n"
    return historical_data

def get_available_years():
    """Get list of available years in the dataset."""
    constructor_standings, races, constructors = load_data()
    seasonal = clean_data(constructor_standings, races, constructors)
    return sorted(seasonal['year'].unique().tolist())

if __name__ == "__main__":
    # Test the functions
    seasonal = get_seasonal_data()
    print(format_historical_data(seasonal))
