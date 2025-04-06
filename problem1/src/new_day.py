import numpy as np
import matplotlib.pyplot as plt
from problem1.src.data_loader import CSVLoader


def extend_binary_array(a, n):
    """
    Extends a binary array 'a' by 'n' elements, preserving its periodic nature
    as best as possible, considering 1s as sparse bits.

    Args:
        a (np.ndarray): The 1D binary numpy array to extend.
        n (int): The number of elements to extend by.

    Returns:
        np.ndarray: The extended binary numpy array. Returns an array of
                    zeros of length len(a) + n if 'a' is considered aperiodic
                    (two or fewer 1s).
    """
    print()
    print(a.astype(int))

    num_ones = np.sum(a)
    if num_ones <= 1:
        return np.zeros(len(a) + n, dtype=int)
    else:
        peaks = []
        for i in reversed(range(len(a))):
            if a[i]:
                peaks.append(i)
            if len(a) == 2:
                break
        step = abs(peaks[1] - peaks[0])
        last_peak = max(peaks)

        if last_peak + step < len(a):
            return np.zeros(len(a) + n, dtype=int)

        extended_array = []
        for i in range(n):
            b = 0
            if (len(a) + i - last_peak) % step == 0:
                b = 1
            extended_array.append(b)

        return np.array(extended_array)


def loss(y_true: np.array, y_pred: np.array) -> float:
    num = (y_true - y_pred) ** 2
    den = y_true + (y_true == 0)
    return float(np.average(num / den))


def model(y_train, n_points=12, window=100):
    """"""
    is_zero_signal = max(y_train) == 0
    signal = 0 if is_zero_signal else np.mean(y_train[y_train > 0])
    threshold = signal / 2

    y_prime = y_train[1:] - y_train[:-1]
    jumps_up = y_prime > threshold
    jumps_down = y_prime < -threshold

    predicted_jumps_up = extend_binary_array(jumps_up[-window:], n_points)
    predicted_jumps_down = extend_binary_array(jumps_down[-window:], n_points)


    y_pred = []
    value = 0 if y_train[-1] == 0 else signal
    for i in range(n_points):
        if predicted_jumps_up[i]:
            value = signal
        elif predicted_jumps_down[i]:
            value = 0
        y_pred.append(value)

    return np.array(y_pred)


def test_continuation(path='..\\..\\data\\01_input_history.csv',
                      n_points=12, threshold=40):

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
    test_continuation()
