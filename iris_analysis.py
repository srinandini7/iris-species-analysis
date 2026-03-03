"""
PROJECT: Iris Botanical Data - Statistical Analysis & Visualization
GOAL: To identify mathematical differences between plant species using 
      hypothesis testing and multi-variable data visualization.
SKILLS: Python (Pandas, Scipy, Seaborn), Numerical Logic (T-Tests), Communication (EDA)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import ttest_ind

# --- 1. DATA PREPARATION ---
# Loading the dataset from sklearn for raw numerical analysis
iris_raw = load_iris()
data = iris_raw.data
target = iris_raw.target

# Converting to a Pandas DataFrame for easier manipulation and "Communication"
# We combine the numerical data with the species names
df = pd.DataFrame(data, columns=iris_raw.feature_names)
df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

print("Dataset Head:\n", df.head())

# --- 2. NUMERICAL APTITUDE: HYPOTHESIS TESTING ---
# We use a T-Test to see if the difference in Sepal Length between 
# two species is statistically significant or just random noise.
group1 = data[target == 0] # Setosa
group2 = data[target == 1] # Versicolor

# Calculate T-Statistic and P-Value
t_stat, pvalue = ttest_ind(group1[:, 0], group2[:, 0])

print(f"\n--- Statistical Significance Test ---")
print(f"T-Statistic: {t_stat:.4f}, P-Value: {pvalue:.4f}")

# Logic: If p-value < 0.05, the difference is not by chance
if pvalue < 0.05:
    print("Conclusion: The difference in sepal length is STATISTICALLY SIGNIFICANT.")
else:
    print("Conclusion: No significant difference found.")

# --- 3. COMMUNICATION: DATA VISUALIZATION ---

# A. Pairplot: The "Big Picture"
# Shows how every feature relates to every other feature, grouped by species
sns.pairplot(df, hue="species", palette="husl")
plt.suptitle("Multi-Variable Feature Correlation", y=1.02)
plt.show()

# B. Boxplot: Showing Distribution & Outliers
# Excellent for seeing the "Spread" and "Median" of data numerically
plt.figure(figsize=(8, 5))
sns.boxplot(x="species", y="sepal length (cm)", data=df)
plt.title("Sepal Length Distribution by Species")
plt.show()

# C. Correlation Heatmap: Numerical Relationships
# Proves aptitude in identifying which variables (e.g., Petal Length/Width) 
# move together mathematically.
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# D. Swarm Plot: Visualizing Every Data Point
# Shows the exact density of the petal width for each species
plt.figure(figsize=(8, 5))
sns.swarmplot(x="species", y="petal width (cm)", data=df)
plt.title("Petal Width Density by Species")
plt.show()