import numpy as np
import matplotlib.pyplot as plt
from problem1.src.data_loader import CSVLoader


def loss(y_true: np.array, y_pred: np.array) -> float:
    num = (y_true - y_pred) ** 2
    den = y_true + (y_true == 0)
    return float(np.average(num / den))


def test_csv_loader(path='..\\..\\data\\01_input_history.csv'):
    """
    Load data test
    Country, Product, Month, Quantity
    """
    # Create an instance of the CSVLoader
    loader = CSVLoader(path)

    # Load the data into a DataFrame
    df = loader.load_data()

    # Print the DataFrame if loading was successful
    print("\nDataFrame loaded successfully!")
    print(df)

    # one-hot
    print("\nOne-hot columns")
    a = loader.get_one_hot_data(columns_to_encode=["Country", "Product"])
    print(list(a.columns))
    # for col in a.columns:
    #     print(type(a[col][0]), col)

    # # load data as pyTorch tensor
    # print("\nTensor")
    # a = loader.get_tensor_data()


def test_preprocessing(path='..\\..\\data\\01_input_history.csv'):

    loader = CSVLoader(path, do_filter_zeros=False)
    loader.load_data(time_years=2.)

    countries = loader.get_countries()
    products = loader.get_products()

    for country in countries:
        for product in products:

            x, y = loader.get_time_series(product, country=country)
            if len(x) == 0:
                continue

            plt.plot(x, y)
            plt.title(f'{product} in {country}')
            plt.show()


def test_delete_pair(path='..\\..\\data\\01_input_history.csv',
                     product="MorningMint", country="Japan"):
    loader = CSVLoader(path, do_filter_zeros=False)
    loader.load_data()
    df = loader.get_dataframe()
    print(len(df))

    loader.delete_rows_by_product_and_country(product, country)
    df = loader.get_dataframe()
    print(len(df))

    print(loader.get_missing_combinations())


def model(y_train, n_points=12, window_sizes=(3, 4, 6, 24, 24), n_windows=4):
    """"""
    # period detector
    mse = np.zeros(len(window_sizes))
    for i, window_size in enumerate(window_sizes):
        y1 = y_train[-(n_windows+1)*window_size:-window_size]
        y2 = y_train[-n_windows*window_size:]
        if np.sum(y1 + y2) == 0:
            mse[i] = np.inf
        else:
            mse[i] = np.mean((y1-y2)**2)

    # print(mse)
    if min(mse) == np.inf:
        best_window = n_points
    else:
        best_window = window_sizes[np.argmin(mse)]

    # copy from median window
    counter = 0
    for i in range(mse.shape[0]):
        if i > 10000000:
            counter += 1
        else:
            break

    if counter != mse.shape[0]:
        n_windows = mse.shape[0] - counter
    windows = [y_train[-best_window:]]
    for i in range(2, n_windows):
        wstart = i * -best_window
        wend = (i-1) * -best_window
        windows.append(y_train[wstart:wend])
#    windows = [y_train[len(y_train)-(i+1)*best_window:len(y_train)-i*best_window] for i in range(n_windows)]
    stacked_array = np.stack(windows, axis=0)
    median_array = np.median(stacked_array, axis=0)  # todo median
    y_pred = median_array
    # y_pred = y_train[-best_window:]

    # while output is shorter than n_points -> repeat
    if y_pred.shape[0] > n_points:
        y_pred = y_pred[:n_points]
    else:
        n_repeats = n_points // y_pred.shape[0]
        y_pred = y_pred.repeat(n_repeats)
    # once it's too long -> cut
#    y_pred = y_pred[:n_points]
    # emergency break
    zero_perc = (median_array == 0).sum() / median_array.shape[0]
    print(zero_perc)
    check_idx=n_points // 2
    if zero_perc > 0.91:
        check_idx*=2
    if np.sum(y_train[-check_idx:]) == 0:
        y_pred *= 0
    print(best_window, y_train, y_pred, median_array)

    # do the mean
    if np.sum(y_pred / (np.max(y_pred)+1e-6) > 0.5) == 0:
        y_pred = np.ones(len(y_pred)) * np.mean(y_pred)

    return y_pred


def test_continuation(path='..\\..\\data\\01_input_history.csv',
                      n_points=12, threshold=30):

    loader = CSVLoader(path, do_filter_zeros=False)
    loader.load_data(time_years=2.)

    countries = loader.get_countries()
    products = loader.get_products()

    costs = []

    for country in countries:
        for product in products:
            # product = "EasyWash Pet Laundry Detergent"
            # country = "Japan"
            x, y = loader.get_time_series(product, country=country)
            x_train = x[:-n_points]
            y_train = y[:-n_points]
            x_pred = x[-n_points:]
            y_true = y[-n_points:]

            # ----- Prediction -----
            y_pred = model(y_train, n_points)
            # ----------------------

            cost = loss(y_true, y_pred)
            costs.append(cost)

            # ----- Plot -----
            if cost > threshold:
                print(f'mean cost = {np.mean(costs):.1f}')

                if len(x) == 0:
                    continue

                plt.plot(x, y)
                plt.plot(x_pred, y_pred)
                plt.title(f'{product} in {country}\n'
                          f'Loss = {cost:.1f}')
                plt.show()

    costs = np.array(costs)
    print()
    print(f'Complete misses = {np.mean(costs>threshold):.1%}')
    print(f'Loss = {np.mean(costs):.0f}')

    plt.hist(np.log10(costs+1))
    plt.show()


if __name__ == '__main__':
    # test_csv_loader()
    # test_preprocessing()
    # test_delete_pair()
    test_continuation()
