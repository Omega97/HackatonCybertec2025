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
    print("DataFrame loaded successfully!")
    print(df)

    # one-hot
    print("one-hot columns")
    a = loader.get_one_hot_data(columns_to_encode=["Country", "Product"])
    print(a.columns)

    # # load data as pyTorch tensor
    print("tensor")
    a = loader.get_tensor_data()
    print(a)


if __name__ == '__main__':
    test_csv_loader()
