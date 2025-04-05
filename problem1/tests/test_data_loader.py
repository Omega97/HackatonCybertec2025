import matplotlib.pyplot as plt
from problem1.src.data_loader import CSVLoader


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

    loader = CSVLoader(path)
    loader.load_data(time_years=2.)

    countries = loader.get_countries()
    products = loader.get_products()

    for country in countries:
        for product in products:

            x, y = loader.get_time_series(product, country=country)
            if len(x) == 0:
                continue

            print(loader.get_time_series_last_days(product, country, time_years=0.1))

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


if __name__ == '__main__':
    # test_csv_loader()
    # test_preprocessing()
    test_delete_pair()
