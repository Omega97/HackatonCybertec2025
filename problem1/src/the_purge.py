from data_loader import CSVLoader
from sklearn import preprocessing
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def main():
    data_path = "./data/01_input_history.csv"
    loader = CSVLoader(data_path, do_filter_zeros=False)
    data = loader.load_data()
    quants = data["Quantity"].to_numpy()
    abs_time = data["abs_time"].to_numpy()

    lep = preprocessing.LabelEncoder()
    lep.fit(data["Product"])
    products = lep.transform(data["Product"])

    lec = preprocessing.LabelEncoder()
    lec.fit(data["Country"])
    countries = lec.transform(data["Country"])

    all_data = []
    for i in range(lec.classes_.shape[0]):
        for j in range(lep.classes_.shape[0]):
            quants_ = quants[np.logical_and(products==j, countries==i)]
            times = abs_time[np.logical_and(products==j, countries==i)]
            all_data.append({
                "country": lec.classes_[i],
                "product": lep.classes_[j],
                "product_int": j,
                "country_int": i,
                "data": quants_,
                "times": times,
            })
    losses = []
    preds = []
    for i in all_data:
        label = i['data'][-12:]
        i["data"] = i['data'][:-12]
        pred = model(i)
        pos_mask = label > 0.00001
        pos_loss = (pred[pos_mask]-label[pos_mask])**2 / label[pos_mask]
        neg_loss = pred[~pos_mask]**2
        losses.append(np.concat([pos_loss, neg_loss]).mean())
        preds.append(pred)

    losses = np.array(losses)
    print("loss: ", losses.mean())

    plot_high_losses(losses, all_data)

def plot_high_losses(losses, all_data):
    high_losses = np.argwhere(losses > 20)[0]
    for i in high_losses:
        fig, ax = plt.subplots()
        labels = all_data[i]["data"][-12:]
        ...

WINDOWS = [4, 6, 12, 24]


def model(data: dict):
    times = data["times"]
    values = data["data"]
    valmax = values.max()
    if valmax == 0:
        return np.zeros(12)

    values = values / valmax
    window_size_sims = []
    best_window_sims = []
    for window_size in WINDOWS:
        n_windows = values.shape[0] // window_size
        all_sims = []
        for i in tqdm.tqdm(range(0, n_windows-1), total=n_windows-1):
            if i == 0:
                current_window = values[-window_size:]
            else:
                current_window = values[(-i-1)*window_size:-i*window_size]
            current_window_sims = []
            for j in range(1, n_windows-1):
                if current_window.sum() < 0:
                    current_window_sims.append(np.inf)
                window = values[-(j+1)*window_size:-j*window_size]
                current_window_sims.append(((window-current_window)**2).mean())
            all_sims.append(np.array(current_window_sims).mean())
        window_size_sims.append(np.array(all_sims).mean())
        best_window_sims.append(np.argmin(np.array(all_sims)))
    window_size = WINDOWS[np.argmin(np.array(window_size_sims))]
    window_idx = best_window_sims[np.argmin(np.array(window_size_sims))]

    if window_idx == 0:
        prediction = values[-window_size:]
    else:
        window_idx += 1
        prediction = values[-window_idx*window_size:-(window_idx-1)*window_size]

    if prediction.shape[0] > 12:
        prediction = prediction[:12]
    else:
        n_repeats = 12 // prediction.shape[0]
        prediction = np.repeat(prediction, n_repeats)

    return prediction

if __name__ == "__main__":
    main()
