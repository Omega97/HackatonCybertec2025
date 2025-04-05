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

japan_morningmint = data[data.isin(["Japan"])]
japan_morningmint.plot.line()
plt.show()

...
