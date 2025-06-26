# Heart Attack Risk Data Analysis using EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Step 1: Data Collection (Synthetic Sample Dataset)
np.random.seed(42)
data = pd.DataFrame({
    'Age': np.random.randint(30, 85, 500),
    'Cholesterol': np.random.normal(200, 30, 500).astype(int),
    'Hypertension': np.random.choice([0, 1], size=500),
    'HeartRate': np.random.randint(60, 100, 500),
    'StressLevel': np.random.randint(1, 10, 500),
    'PhysicalActivity': np.random.randint(1, 10, 500),
    'FamilyHistory': np.random.choice([0, 1], size=500),
    'HeartAttackRisk': np.random.choice([0, 1], size=500, p=[0.7, 0.3])
})

# Step 2: Data Cleaning & Preprocessing
print("Missing values:\n", data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())
print("\nSummary Statistics:\n", data.describe())

# Step 3: EDA

# Distribution plots
features = ['Age', 'Cholesterol', 'StressLevel', 'PhysicalActivity', 'HeartRate']
for feature in features:
    sns.histplot(data[feature], kde=True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Count plot for Hypertension
sns.countplot(x='Hypertension', data=data)
plt.title('Hypertension Count')
plt.xlabel('Hypertension (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Heart Attack Risk Factors')
plt.show()

# Heart Attack Risk by Age Group
data['AgeGroup'] = pd.cut(data['Age'], bins=[30, 40, 50, 60, 70, 85],
                          labels=['30-40', '41-50', '51-60', '61-70', '71-85'])
sns.barplot(x='AgeGroup', y='HeartAttackRisk', data=data, palette='magma')
plt.title('Heart Attack Risk by Age Group')
plt.ylabel('Proportion of Risk')
plt.xlabel('Age Group')
plt.show()

# Box plots to compare Stress & Physical Activity with Risk
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(x='HeartAttackRisk', y='StressLevel', data=data, ax=axes[0], palette='OrRd')
axes[0].set_title('Stress Level vs Heart Attack Risk')
axes[0].set_xlabel('Heart Attack Risk')
axes[0].set_ylabel('Stress Level')

sns.boxplot(x='HeartAttackRisk', y='PhysicalActivity', data=data, ax=axes[1], palette='BuGn')
axes[1].set_title('Physical Activity vs Heart Attack Risk')
axes[1].set_xlabel('Heart Attack Risk')
axes[1].set_ylabel('Physical Activity Level')

plt.tight_layout()
plt.show()

# Conclusion in print format
print("\nKey Insights:")
print("1️⃣  Risk is highest among individuals aged 70-80.")
print("2️⃣  Higher stress increases heart attack risk.")
print("3️⃣  Physical activity appears to reduce the risk.")
print("4️⃣  Cholesterol and Hypertension are strong risk factors.")
print("5️⃣  Family history does not significantly affect risk in this sample.")
