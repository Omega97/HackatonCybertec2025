from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch
from data_loader import CSVLoader
from sklearn import preprocessing
import numpy as np
from pathlib import Path
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, val: bool=False, val_year=2023, return_samp=False):
        self.return_samp = return_samp
        self.val = val
        self.val_year = val_year
        self.quants = data["Quantity"].to_numpy()
        self.abs_time = data["abs_time"].to_numpy()

        lep = preprocessing.LabelEncoder()
        lep.fit(data["Product"])
        products = lep.transform(data["Product"])

        lec = preprocessing.LabelEncoder()
        lec.fit(data["Country"])
        countries = lec.transform(data["Country"])

        self.data = []
        for i in range(lec.classes_.shape[0]):
            for j in range(lep.classes_.shape[0]):
                data = self.quants[np.logical_and(products==i, countries==j)]
                times = self.abs_time[np.logical_and(products==i, countries==j)]
                if data.shape[0] == 0:
                    continue
                self.data.append({
                    "country": lec.classes_[i],
                    "product": lep.classes_[j],
                    "product_int": j,
                    "country_int": i,
                    "data": torch.tensor(data),
                    "times": torch.tensor(times - 2000)
                })

    @staticmethod
    def remove_zero_vals(values, positions):
        first_nonzero = torch.where(values > 0)[0][0]
        if positions[-1] - positions[first_nonzero] < 2:
            ...
        values = values[first_nonzero:]
        positions = positions[first_nonzero:]
        return values, positions

    def __getitem__(self, index):
        sample = self.data[index]
        values = sample["data"]
        positions = sample["times"]
        values, positions = self.remove_zero_vals(values, positions)
        mask = torch.zeros_like(values, dtype=torch.bool)
        if self.val:
            mask[positions >= self.val_year-2000] = True
        else:
            # pick random start point between 2004 and 2022:
            period = 1
            start = positions.min() + period
            end = positions.max() - 1
            start = torch.FloatTensor(1).uniform_(start.item(), end.item())
            mask_mask = torch.logical_and(positions >= start, positions < start+1)
            mask[mask_mask] = True
            new_start = start - period
            first_true = torch.where(positions > new_start)[0][0]
            last_true = torch.where(mask)[0][-1]
            mask = mask[first_true:last_true]
            values = values[first_true:last_true]
            values = values + torch.randint(-15, 15, size=(values.shape))
            values[values < 0] = 0
            positions = positions[first_true:last_true]

        sample = {
            "values": values[..., None, None, None].float(),
            "mask": mask,
            "positions": positions[None, ...].float(),
            "country": sample["country_int"],
            "product": sample["product_int"],
        }
        if self.return_samp:
            sample["samp"] = self.data[index]
        return sample

    def __len__(self):
        return len(self.data)


def load_dataset():
    dataset_filepath = "./data/01_input_history.csv"
    dataset_cache = Path("./cached_dataset.pkl")
    if not dataset_cache.exists():
        data = CSVLoader(
            file_path=dataset_filepath,
        ).load_data(time_years=1)
        data.to_csv(dataset_cache)
    else:
        data = pd.read_csv(dataset_cache, index_col=0)

    validation = data[data["year_int"] >= 2022]
    train = data[data["year_int"] <= 2022]
    val_ds = Dataset(data=validation, val=True)
    train_ds = Dataset(data=train)
    return val_ds, train_ds


def main():
    val, train = load_dataset()
    encoder_config = get_encoder_size("RoMA-tiny")
    model_config = RoMAForPreTrainingConfig(
        encoder_config=encoder_config,
        decoder_config=encoder_config,
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=1
    )
    model = RoMAForPreTraining(model_config)
    trainer_config = TrainerConfig(
        base_lr=3e-3,
        epochs=1600,
        warmup_steps=500,
        eval_every=200,
        save_every=200,
        batch_size=128,
        optimizer="adamw",
        optimizer_args={"weight_decay": 3e-4, "betas": (0.9, 0.999)},
        project_name="Problem 1",
        entity_name=None
    )
    trainer = Trainer(trainer_config)
    trainer.train(
        model=model,
        train_dataset=train,
        test_dataset=val
    )

if __name__ == "__main__":
    main()