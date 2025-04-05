from problem1.src.data_loader import CSVLoader


def test_prediction(training_path='..\\..\\data\\01_input_history.csv',
                    test_path='..\\..\\data\\01_output_prediction_example.csv'):
    training_df = CSVLoader(training_path, do_filter_zeros=False)
    training_df.load_data()

    output_df = CSVLoader(test_path, do_filter_zeros=False)
    output_df.load_data()

    df_in = training_df.get_dataframe()
    df_reference = output_df.get_dataframe()
    df_out = df_reference.copy()

    v = []
    for index, row in df_out.iterrows():
        # print(f"{index} {row}")
        product = row["Product"]
        country = row["Country"]
        x, y = training_df.get_time_series(product, country)
        prediction = 0
        v.append(prediction)

    # df_out = df_out['col1'].apply(v)
    print(df_out)


if __name__ == '__main__':
    test_prediction()
