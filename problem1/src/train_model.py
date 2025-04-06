import random
from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch
from data_loader import CSVLoader
from sklearn import preprocessing
import numpy as np
from pathlib import Path
import torch.nn as nn
import wandb


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
                quants = torch.tensor(self.quants[np.logical_and(products==j, countries==i)].copy(), dtype=torch.float)
                times = torch.tensor(self.abs_time[np.logical_and(products==j, countries==i)].copy(), dtype=torch.float)
                quants, times = self.remove_zero_vals(quants, times)
                if quants.shape[0] == 0:
                    continue
                qmax = quants.max()
                quants = quants / qmax
                self.data.append({
                    "country": lec.classes_[i],
                    "product": lep.classes_[j],
                    "product_int": j,
                    "country_int": i,
                    "data": quants,
                    "times": times - 2000,
                    "max": qmax
                })

    def remove_zero_vals(self, values, positions):
        if values.sum() < 0.01:
            return torch.tensor([]), torch.tensor([])
        nonzeros = torch.where(values > 0)[0]
        first_nonzero = nonzeros[0]

        if self.val:
            values[:first_nonzero] = values.mean()
            return values, positions

        last_nonzero = nonzeros[-1] + 1
        values = values[first_nonzero:last_nonzero]
        positions = positions[first_nonzero:last_nonzero]
        if (positions-positions.min()).max() < 2:
            return torch.tensor([]), torch.tensor([])
        return values, positions

    def __getitem__(self, index):
        sample = self.data[index]
        values = sample["data"]
        positions = sample["times"]
        mask = torch.zeros_like(values, dtype=torch.bool)
        if self.val:
            mask[positions >= self.val_year-2000] = True
        else:
            # pick random start point between 2004 and 2022:
            period = 1
            start = positions.min() + period
            end = positions.max() - 1
            start = torch.FloatTensor(1).uniform_(start.item(), end.item())
            mask_mask = torch.logical_and(positions >= start, positions <= start+1)
            new_start = start - period
            first_true = torch.where(positions > new_start)[0][0]
            last_true = first_true + 24
            mask[first_true+12:last_true] = True
            mask = mask[first_true:last_true]
            values = values[first_true:last_true]
            values = values + torch.zeros_like(values).normal_(std=0.1)
            values[values < 0] = 0
            positions = positions[first_true:last_true]
            if torch.rand((1, 1)) > 0.5:
                rand_idx = torch.randint(low=0, high=10, size=(1, 1)).item()
                values[rand_idx:rand_idx+2] = 0
                values[rand_idx+12:rand_idx+14] = 0
            else:
                rand_idx = torch.randint(low=0, high=3, size=(1, 1)).item()
                values[:rand_idx] = 0

        sample = {
            "values": values[..., None, None, None].float(),
            "mask": mask,
            "positions": positions[None, ...].float(),
            "country": sample["country_int"],
            "product": sample["product_int"],
            "max": sample["max"]
        }
        if self.return_samp:
            sample["samp"] = self.data[index]
        return sample

    def __len__(self):
        return len(self.data)


def eval_callback(model, loss_train, test_dataloader):
    for batch in test_dataloader:
        batch = {key: val.to(loss_train.device) for key, val in batch.items()}
        logits, _ = model(**batch)
        break
    logits = logits.detach().cpu().numpy()
    labels = batch["values"].detach().cpu().numpy()
    positions = batch["positions"].detach().cpu().numpy()

    logits = logits.squeeze()[:, :12, None]
    mask = nn.functional.sigmoid(logits) > 0.5
    logits[mask] = batch["labels"][mask]
    logits[~mask] = 0
    for i in range(16):
        logit = logits[i].squeeze() * batch["max"][i].item()
        logit_quant = logit[:, 0]
        logit_bin = logit[:, 1]
        logit_quant[logit_bin < 0.5] = 0
        logit = logit_quant
        label = labels[i].squeeze() * batch["max"][i].item()
        position = positions[i].squeeze()
        logit = np.concat([label[:12], logit])

        wandb.log({f"ModelPredictions-{i}" : wandb.plot.line_series(
            xs=position.tolist(),
            ys=[logit.tolist(), label.tolist()],
            keys=["logit", "label"],
            title="Logit vs label",
            xname="time")})


def load_dataset():
    dataset_filepath = "./data/01_input_history.csv"
    dataset_cache = Path("./cached_dataset.pkl")
#    if not dataset_cache.exists():
    data = CSVLoader(
        file_path=dataset_filepath, do_filter_zeros=False
    ).load_data(time_years=1)
#    else:
#        data = pd.read_csv(dataset_cache, index_col=0)

    validation = data[data["year_int"] >= 2022]
    train = data[data["year_int"] <= 2022]
    val_ds = Dataset(data=validation, val=True)
    train_ds = Dataset(data=train)
    return val_ds, train_ds


class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, label, og_x):
        og_x = og_x.squeeze()[:, :12, None]
        mask = nn.functional.sigmoid(logits) > 0.5
        logits[mask] = og_x[mask]
        logits[~mask] = 0
        true_mask = label > 0.001
        loss_pos = (logits[true_mask]-label[true_mask])**2/label[true_mask]
        loss_neg = logits[~true_mask]**2
        np = loss_pos.shape[0] + loss_neg.shape[0]
        return (loss_pos.sum() + loss_neg.sum()) / np


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
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
    model.set_loss_fn(CustomLoss())
    trainer_config = TrainerConfig(
        base_lr=7e-4,
        epochs=400,
        warmup_steps=400,
        eval_every=200,
        save_every=200,
        batch_size=128,
        optimizer="adamw",
        optimizer_args={"weight_decay": 3e-4, "betas": (0.9, 0.999)},
        project_name="Problem 1",
        entity_name=None
    )
    trainer = Trainer(trainer_config)
    trainer.set_post_train_hook(eval_callback)
    trainer.train(
        model=model,
        train_dataset=train,
        test_dataset=val
    )

if __name__ == "__main__":
    main()