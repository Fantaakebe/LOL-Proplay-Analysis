League of Legends Proplay Analysis

Description:

This project analyzes professional League of Legends matches to identify key metrics that contribute to team success. 
Using data from Oracle's Elixir, the project evaluates various factors, such as kills, assists, deaths, gold earned, 
and champion choices, to determine their impact on the outcome of games. The goal is to help players and teams understand what aspects of player performance are most influential in winning professional matches.

## Table of Contents
- [Introduction](#introduction)
- [Data Source](#data-source)
- [Methods Used](#methods-used)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributors](#contributors)
- [Future Work](#future-work)
- [License](#license)

Introduction:

League of Legends is a global multiplayer online battle arena (MOBA) game with over 100 million users. 
In this project, we analyze professional esports matches from 2014 to 2024, focusing on factors that influence team success, 
such as player performance metrics (kills, deaths, assists) and champion picks. 
Our findings can help aspiring pro players and analysts understand what it takes to excel in competitive play.

Data Source:

The data used in this project is sourced from Oracle's Elixir**[Oracle's Elixir](https://oracleselixir.com/tools/downloads). 

The dataset includes:

- Player performance metrics (kills, deaths, assists, etc.)
- Champion picks and bans
- Match outcomes
The data is collected from a variety of official Riot APIs and publicly available esports match histories.

Methods Used:

The project employs several data science techniques to analyze the data:

1. Linear Regression: To evaluate the impact of various player performance metrics on team success.
- Metrics analyzed:
  kills,
  deaths,
  assists,
  total gold,
  damage to champions,
  and vision score.

3. Visualizations:
- Line charts to show champion popularity trends over time.
- Bubble charts to analyze champion win rates and their pick frequency.
- Heatmaps to explore champion picks by region over the years.

3. Prediction Models:
- ROC curve to measure model performance in predicting match outcomes.
- Top 20 features to identify the most significant factors influencing match outcomes, including gold difference and experience difference at 15 minutes.

Installation:

To run this project on your local machine:

Clone the repository:
bash
Copy code
git clone https://github.com/Fantaakebe/LOL-Proplay-Analysis.git
Navigate to the project directory:
bash
Copy code
cd LOL-Proplay-Analysis
Install the necessary Python packages:
bash
Copy code
pip install -r requirements.txt

Usage

Run the analysis scripts:
Use LOL_prediction_analysis.py to execute predictive models on match data.
Open champion_visualizations.ipynb in Jupyter Notebook to explore visualizations of champion trends.
Customize the analysis by downloading updated datasets from Oracle's Elixir and modifying the analysis scripts to work with new data.

Features:
- Player Performance Analysis: Analyze how metrics like kills, deaths, and assists affect team success using linear regression.
- Champion Popularity Trends: Visualize changes in champion picks over time and across different regions.
- Predictive Modeling: Evaluate which in-game factors (e.g., gold difference, XP difference) are the strongest predictors of match outcomes.


Contributors:

Jaden Chin - GitHub   
Rukia Nur - GitHub     
Fanta Kebe - [GitHub Profile](https://github.com/Fantaakebe)

Future Work:

- Deeper Champion Analysis: Conduct in-depth analyses on individual champions to understand their performance in different scenarios.
- General Player Data: Expand the analysis to include data from casual and semi-professional players to see how playstyles differ by skill level.
- Machine Learning Models: Further improve predictive modeling by applying more advanced machine learning techniques to enhance the accuracy of predictions.
