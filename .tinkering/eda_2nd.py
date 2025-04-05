import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/home/l11/Desktop/HackatonCybertec2025/data/02_input_target.csv')

# 1. Basic Exploration
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Distribution Analysis
# For numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# 3. Time Series Analysis (if applicable)
# Assuming 'Month' is a datetime column
if 'Month' in df.columns:
    df['Month'] = pd.to_datetime(df['Month'])
    plt.figure(figsize=(12, 6))
    df.groupby('Month')['Quantity'].sum().plot()
    plt.title('Total Quantity Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Quantity')
    plt.show()

# 4. Correlation Analysis
if len(numerical_cols) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# 5. Categorical Analysis
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Count of {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=col, y='Quantity', data=df)
    plt.title(f'Quantity by {col}')
    plt.xticks(rotation=45)
    plt.show()

# 6. Advanced Analysis
# Country-Product Analysis
if 'Country' in df.columns and 'Product' in df.columns:
    country_product = df.groupby(['Country', 'Product'])['Quantity'].sum().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(country_product, cmap='YlGnBu', annot=True, fmt='g')
    plt.title('Quantity by Country and Product')
    plt.show()

# 7. Outlier Detection
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Outlier Detection for {col}')
    plt.show()

# 8. Trend Analysis
if 'Month' in df.columns:
    for col in ['Country', 'Product']:
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            for value in df[col].unique():
                temp_df = df[df[col] == value]
                temp_df.groupby('Month')['Quantity'].sum().plot(label=value)
            plt.title(f'Quantity Trend by {col}')
            plt.xlabel('Month')
            plt.ylabel('Quantity')
            plt.legend()
            plt.show()
