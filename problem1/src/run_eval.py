from train_model import Dataset
from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig
from roma.utils import load_from_checkpoint
from data_loader import CSVLoader
import pandas as pd
import torch
import csv
from pathlib import Path


def load_dataset():
    dataset_filepath = "./data/01_input_history.csv"
    dataset_cache = Path("./cached_dataset_eval.pkl")
    if not dataset_cache.exists():
        data = CSVLoader(
            file_path=dataset_filepath,
        ).load_data(time_years=1)
        data.to_csv(dataset_cache)
    else:
        data = pd.read_csv(dataset_cache)

    data = data[data["year_int"] >= 2023]
    data_copy = data.copy()
    data_copy["year_int"] += 1
    data_copy["abs_time"] += 1
    data = pd.concat([data, data_copy])
    ds = Dataset(data=data, val=True, val_year=2024, return_samp=True)
    return ds

def main():
    checkpoint = "./checkpoints/checkpoint-1400"
    model = load_from_checkpoint(checkpoint,RoMAForPreTraining, RoMAForPreTrainingConfig)
    ds = load_dataset()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1)
    header = ["Country", "Product","Month","Quantity"]
    dates = ["Jan2024","Feb2024", "Mar2024","Apr2024","May2024","Jun2024","Jul2024", "Aug2024","Sep2024","Oct2024","Nov2024","Dec2024"]
    output_list = []
    for batch in dataloader:
        samp = batch.pop("samp")
        pred, _ = model(**batch)
        pred = pred.squeeze()
        pred[pred < 0] = 0
        for i in range(12):
            output_list.append((samp["country"][0], samp["product"][0], dates[i], int(pred[i].item())))

    with open("./predictions.csv", "w") as f:
        f.writelines([",".join(header)+"\n"])
        writer = csv.writer(f)
        for i in output_list:
            writer.writerow(i)


if __name__ == "__main__":
    main()