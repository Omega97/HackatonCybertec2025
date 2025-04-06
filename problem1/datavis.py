import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import CSVLoader
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.style.use('seaborn-v0_8')

data_filepath = "./data/01_input_history.csv"
data = CSVLoader(file_path=data_filepath).load_data(time_years=1)
#data = pd.read_csv(data_filepath)

print(data.head())

fig, ax = plt.subplots()
data["Quantity"].hist()
fig.suptitle("Quantities")
fig.show()

print("Countries: ", data["Country"].unique())
print("Products: ", data["Product"].unique())
print("Products size: ", data["Product"].unique().shape[0])
#print("Months: ", data["Month"].unique())

mask = data[["Country", "Product"]].isin(["Japan", "MorningMint"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan MorningMint Quantities", x=data["abs_time"][mask])
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "SoftStep Baby Shampoo"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan SoftStep Baby Shampoo Quantities", x=data["abs_time"][mask])
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "PurePore Toner"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan PurePore Toner Quantities", x=data["abs_time"][mask])
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "LuminousLocks Shampoo"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan LuminousLocks Shampoo Quantities", x=data["abs_time"][mask])
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "PurePore Insect Repellent Spray"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan PurePore Insect Repellent Spray Quantities", x=data["abs_time"][mask])
plt.show()


mask = data[["Country", "Product"]].isin(["Japan", "GentleGlow Dish Soap"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan GentleGlow Dish Soap Quantities")
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "CleanSlate Disinfectant Spray"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Japan CleanSlate Disinfectant Spray Quantities")
plt.show()

mask = data[["Country", "Product"]].isin(["Australia", "RadiantRose Hair Serum"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Australia RadiantRose Hair Serum Quantities")
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "Exfoliating Essentials Scrub"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line(title="Exfoliating Essentials Scrub")
plt.show()