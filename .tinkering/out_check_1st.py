import pandas as pd

# Load the CSV file
file_path = '/home/l11/Desktop/HackatonCybertec2025/data/01_output_prediction_example.csv'
df = pd.read_csv(file_path)

# Get unique countries and products
unique_countries = df['Country'].unique()
unique_products = df['Product'].unique()

# Generate all possible combinations
all_combinations = [(country, product) for country in unique_countries for product in unique_products]

# Check which combinations exist in the data
existing_combinations = set(zip(df['Country'], df['Product']))
missing_combinations = [comb for comb in all_combinations if comb not in existing_combinations]

# Print results
print(f"Total unique countries: {len(unique_countries)}")
print(f"Total unique products: {len(unique_products)}")
print(f"Total possible combinations: {len(all_combinations)}")
print(f"Existing combinations: {len(existing_combinations)}")
print(f"Missing combinations: {len(missing_combinations)}")

if missing_combinations:
    print("\nMissing combinations:")
    for country, product in missing_combinations:
        print(f"{country} - {product}")
else:
    print("\nAll possible combinations exist in the data.")
