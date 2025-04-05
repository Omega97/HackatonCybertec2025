import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_filepath = "./data/01_input_history.csv"

data = pd.read_csv(data_filepath)

print(data.head())

data["Quantity"].hist()
plt.show()

print("Countries: ", data["Country"].unique())
print("Products: ", data["Product"].unique())
print("Products size: ", data["Product"].unique().shape[0])
print("Months: ", data["Month"].unique())

mask = data[["Country", "Product"]].isin(["Japan", "MorningMint"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line()
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "SoftStep Baby Shampoo"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line()
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "PurePore Toner"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line()
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "LuminousLocks Shampoo"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line()
plt.show()

mask = data[["Country", "Product"]].isin(["Japan", "PurePore Insect Repellent Spray"]).all(axis=1)
japan_morningmint = data["Quantity"][mask]
japan_morningmint.plot.line()
plt.show()
