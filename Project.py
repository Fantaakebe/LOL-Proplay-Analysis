"""
Author        : Fanta Kebe
Course        : DS2500: Intermediate Programming with Data
Filename      : Project.py
   
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# File paths for the datasets
file_2021 = r"C:\Users\thefa\Dropbox\PC\Desktop\DS2500\Project\LOL csv files\2021_LoL_esports_match_data_from_OraclesElixir.csv"
file_2022 = r"C:\Users\thefa\Dropbox\PC\Desktop\DS2500\Project\LOL csv files\2022_LoL_esports_match_data_from_OraclesElixir.csv"
file_2023 = r"C:\Users\thefa\Dropbox\PC\Desktop\DS2500\Project\LOL csv files\2023_LoL_esports_match_data_from_OraclesElixir.csv"
file_2024 = r"C:\Users\thefa\Dropbox\PC\Desktop\DS2500\Project\LOL csv files\2024_LoL_esports_match_data_from_OraclesElixir.csv"

# Function to load all datasets
def load_all_data(file_2021, file_2022, file_2023, file_2024):
    dataframes = []
    files = [file_2021, file_2022, file_2023, file_2024]
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Function to preprocess data
def preprocess_data(data):
    features = ['kills', 'deaths', 'assists', 'totalgold', 'damagetochampions', 'visionscore']
    target = 'result'

    # Convert champion column to categorical and create dummy variables
    champions = pd.get_dummies(data['champion'], prefix='champion') 
    # From Stackoverflow: Creating dummy variables in pandas for python
    X = data[features].join(champions)
    y = data[target]

    # Handle missing values by dropping rows with any missing value
    X = X.dropna()
    y = y[X.index]

    return X, y

# Function for regression analysis
def perform_regression_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    coefficients = model.coef_
    intercept = model.intercept_

    return r2, coefficients, intercept, model

# Function for time series analysis on champion popularity
def time_series_analysis_champion_popularity(data):
    # Group by patch and champion to get the pick count for each champion per patch
    champion_data = data.groupby(['patch', 'champion']).size().reset_index(name='pick_count')
    
    # Convert 'patch' column to a categorical type if it's not already
    champion_data['patch'] = pd.Categorical(champion_data['patch'], ordered=True)
    
    # Sort by patch to ensure time progression
    champion_data = champion_data.sort_values('patch')

    # Get the top 10 most picked champions across all patches
    top_champions = champion_data.groupby('champion')['pick_count'].sum().nlargest(10).index
    filtered_data = champion_data[champion_data['champion'].isin(top_champions)]
    
    plt.figure(figsize=(14, 7))
    
    for champion in top_champions:
        champ_data = filtered_data[filtered_data['champion'] == champion]
        
        # Plot the pick counts over time
        plt.plot(champ_data['patch'], champ_data['pick_count'], label=champion)
        
        # Calculate and print the trend over time
        if len(champ_data['pick_count']) > 1:
            trend = champ_data['pick_count'].iloc[-1] - champ_data['pick_count'].iloc[0]
            if trend > 0:
                print(f'{champion} is becoming more popular over time.')
            elif trend < 0:
                print(f'{champion} is becoming less popular over time.')
            else:
                print(f'{champion} popularity has remained stable over time.')

    plt.xlabel('Patch')
    plt.ylabel('Pick Count')
    plt.title('Time Series Analysis of Top 10 Champion Pick Rate Over Time')
    plt.legend()
    plt.show()

def main():
    data = load_all_data(file_2021, file_2022, file_2023, file_2024)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Perform regression analysis
    r2, coefficients, intercept, model = perform_regression_analysis(X, y)
    print(f'R^2 Score: {r2}')
    print(f'Coefficients: {coefficients}')
    print(f'Intercept: {intercept}')
    
    # Print feature names with coefficients
    feature_names = X.columns
    for feature, coef in zip(feature_names, coefficients):
        print(f'{feature}: {coef}')
    
    # Perform time series analysis on champion popularity
    time_series_analysis_champion_popularity(data)

if __name__ == "__main__":
    main()
