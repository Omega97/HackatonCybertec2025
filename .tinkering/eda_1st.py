import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/home/l11/Desktop/HackatonCybertec2025/data/01_input_history.csv')

# Convert Month to datetime
df['Month'] = pd.to_datetime(df['Month'])

# 1. Basic Info
print("Basic Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Time Range Analysis
print("\nTime Range:")
print(f"Start: {df['Month'].min()}")
print(f"End: {df['Month'].max()}")

# 3. Product Analysis (Modified for readability)
product_stats = df.groupby('Product').agg(
    first_month=('Month', 'min'),
    last_month=('Month', 'max'),
    total_quantity=('Quantity', 'sum')
).sort_values('first_month')

print("\n=== Product Start/End Dates ===")
print(product_stats.to_string())

# 4. Seasonal Analysis
df['year'] = df['Month'].dt.year
df['month'] = df['Month'].dt.month

seasonal_analysis = df.groupby(['Product', 'month'])['Quantity'].mean().unstack()
seasonal_analysis.plot(kind='line', figsize=(15,8), title='Monthly Demand Patterns')
plt.show()

# 5. Production Cessation Analysis (Modified for readability)
current_year = df['Month'].max().year
ceased_products = product_stats[product_stats['last_month'].dt.year < current_year]
print("\n=== Potentially Ceased Products ===")
print(ceased_products.to_string())
