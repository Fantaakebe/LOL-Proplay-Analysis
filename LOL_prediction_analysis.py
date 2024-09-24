# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


# Initialize df to hold data
files = glob.glob('*.csv')

df = pd.DataFrame()

# Define the years of interest (past 3 years)
years_of_interest = [2022, 2023, 2024]

dtype_dict = {
    'split': str,
    # You can add other columns here if needed
}

# Process files in chunks, filtering by year
for file in files:
    chunk_iter = pd.read_csv(file, chunksize=10000, low_memory=False)
    for chunk in chunk_iter:
        chunk_filtered = chunk[chunk['year'].isin(years_of_interest)]
        df = pd.concat([df, chunk_filtered], ignore_index=True)
       
# pre-game factors such as champion picks, historical performance, and team stats
features = [
    'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
    'goldat10', 'xpat10', 'csat10', 'golddiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10',
    'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15'
]

X = df[features]
y = df['result']

#missing values
X = X.fillna(0)

# C0nvert categorical variables into dummies
X = pd.get_dummies(X, columns=['champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#randomforest model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate probabilities and ROC curve
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('ROCCurve')
plt.show()


# Get feature importances from the trained eandomforest model
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

feature_names = X.columns

# Plot top 20 feature
plt.figure(figsize=(12, 8))
plt.title("Top 20 Features")
plt.barh(range(20), importances[indices[:20]], align="center")
plt.yticks(range(20), [feature_names[i] for i in indices[:20]])
plt.gca().invert_yaxis()  
plt.xlabel('Relative Impact')
plt.savefig('Top20Features')
plt.show()

# Calculate the win rate for each champion
champion_win_rate = df.groupby('champion')['result'].mean()

# Sort the champions by win rate
champion_win_rate_sorted = champion_win_rate.sort_values(ascending=False)

# Plot the top 20 champions by win rate
plt.figure(figsize=(14, 8))
champion_win_rate_sorted.head(20).plot(kind='bar', color='green')
plt.title('Top 20 Champions by Win Rate')
plt.xlabel('Champion')
plt.ylabel('Win Rate')
plt.xticks(rotation=45)
plt.ylim(0.4, 0.6)
plt.savefig('ChampsbyWin')
plt.show()

