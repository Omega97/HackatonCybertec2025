from data_loader import CSVLoader
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import csv
from new_day import model


def main():
    data_path = "./data/01_input_history.csv"
    loader = CSVLoader(data_path, do_filter_zeros=False)
    data = loader.load_data()
    quants = data["Quantity"].to_numpy()
    abs_time = data["abs_time"].to_numpy()
    months = data["month_int"].to_numpy()

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
            months_ = months[np.logical_and(products==j, countries==i)]
            all_data.append({
                "country": lec.classes_[i],
                "product": lep.classes_[j],
                "product_int": j,
                "country_int": i,
                "data": quants_,
                "times": times,
                "month": months_
            })
    losses = []
    preds = []
    for idx, i in enumerate(all_data):
        if idx == 314:
            ...
        label = i['data'][-12:]
        i["og_data"] = i["data"]
        i["data"] = i['data'][:-12]
        pred = model(i)
        pos_mask = label > 0.00001
        pos_loss = (pred[pos_mask]-label[pos_mask])**2 / label[pos_mask]
        neg_loss = pred[~pos_mask]**2
        losses.append(np.concat([pos_loss, neg_loss]).mean())
        preds.append(pred)

    losses = np.array(losses)
    print("loss: ", losses.mean())
    save_predictions("./01_output_prediction_4095.csv", preds, all_data)
    plot_high_losses(losses, all_data, preds)

def save_predictions(filepath, preds, all_data):
    header = ["Country","Product","Month","Quantity"]
    times = ["Jan2024",  "Feb2024", "Mar2024", "Apr2024", "May2024", "Jun2024",
             "Jul2024", "Aug2024", "Sep2024", "Oct2024", "Nov2024", "Dec2024"]
    processed_data = []
    for i, j in zip(preds, all_data):
        for k in range(i.shape[0]):
            processed_data.append(
                {"Country": j["country"],
                 "Product": j["product"],
                 "Month": times[j["month"][k]-1],
                 "Quantity": int(i[k])})

    with open(filepath, "w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(processed_data)

def plot_high_losses(losses, all_data, preds):
    high_losses = np.argwhere(losses > 50)
    for i in high_losses[:20]:
        i = i[0]
        fig, ax = plt.subplots()
        labels = all_data[i]["og_data"]
        times = all_data[i]["times"]
        pred = preds[i]
        ax.plot(times, labels)
        ax.plot(times[-12:], pred)
        fig.suptitle(f"Loss: {losses[i]}, Item: {i}")
        fig.show()

WINDOWS = [4, 6, 12, 24]


def model_(data: dict):
    times = data["times"]
    values = data["data"]
    valmax = values.max()
    n_windows = 4
    n_points = 12
    mse = np.zeros(len(WINDOWS))
    for i, window_size in enumerate(WINDOWS):
        y1 = values[-2*window_size:-window_size]
        y2 = values[-window_size:]
        if np.sum(y1 + y2) == 0:
            mse[i] = np.inf
        else:
            mse[i] = np.mean((y1-y2)**2)

    if min(mse) == np.inf:
        best_window = n_points
    else:
        best_window = WINDOWS[np.argmin(mse)]

    windows = [values[-best_window:]]
    for i in range(2, n_windows+1):
        wstart = i * -best_window
        wend = (i-1) * -best_window
        wind = values[wstart:wend]
        if wind.sum() > 0:
            windows.append(wind)

    stacked_array = np.stack(windows, axis=0)
    median_array = np.median(stacked_array, axis=0)  # todo median
    y_pred = median_array

    if y_pred.shape[0] > n_points:
        y_pred = y_pred[:n_points]
    else:
        n_repeats = n_points // y_pred.shape[0]
        y_pred = np.concat([y_pred for i in range(n_repeats)], axis=0)

    zero_perc = (median_array == 0).sum() / median_array.shape[0]
    check_idx=6
    if zero_perc > 0.93:
        check_idx*=4
    elif np.sum(values[-check_idx:]) == 0:
        y_pred *= 0

    # do the mean
    if np.sum(y_pred / (np.max(y_pred)+1e-6) > 0.5) == 0:
        y_pred = np.ones(len(y_pred)) * np.mean(y_pred)

    return y_pred


if __name__ == "__main__":
    main()
