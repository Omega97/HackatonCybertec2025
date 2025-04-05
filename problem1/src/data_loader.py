import pandas as pd
import torch


class CSVLoader:
    """
    A class to read data from a CSV file, load it into a Pandas DataFrame,
    perform one-hot encoding on specified columns, and convert it to a PyTorch tensor.
    """
    def __init__(self, file_path, time_column='Month', time_format="%b%Y"):
        """
        Initializes the CSVLoader with the path to the CSV file.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.time_column = time_column
        self.time_format = time_format
        self.dataframe = None
        self.one_hot_dataframe = None
        self.tensor_data = None

    def _preprocess_time(self):
        """
        Changes the specified time column to two new columns:
        "month_int" (integer representation of the month)
        "year_int" (year as integer)
        "abs_time" absolute time
        """
        if self.dataframe is not None and self.time_column in self.dataframe.columns:
            try:
                # Convert the time column to datetime objects
                self.dataframe[self.time_column] = pd.to_datetime(self.dataframe[self.time_column],
                                                                  format=self.time_format)

                # Create new columns for month (integer) and year (integer)
                self.dataframe['month_int'] = self.dataframe[self.time_column].dt.month.astype(int)
                self.dataframe['year_int'] = self.dataframe[self.time_column].dt.year.astype(int)
                self.dataframe['abs_time'] = self.dataframe['year_int'] + (self.dataframe['month_int']-1)/12

                # Optionally drop the original time column if needed
                self.dataframe = self.dataframe.drop(columns=[self.time_column])
            except KeyError:
                print(f"Error: Time column '{self.time_column}' not found in the DataFrame.")
            except ValueError:
                print(
                    f"Error: Could not parse time values in column '{self.time_column}' using format '{self.time_format}'. Ensure '{self.time_format}' correctly matches your time string (e.g., '%b%Y' for 'Feb2011').")
            except Exception as e:
                print(f"An error occurred during time preprocessing: {e}")
        elif self.time_column:
            print("Warning: DataFrame not loaded or time column not found for preprocessing.")

    def delete_rows_by_country(self, product, country):
        """
        Deletes all rows with that (product, country) couple.
        """
        df =  self.get_dataframe()

        # Create a boolean mask where True indicates rows to KEEP
        mask = ~((df['Country'] == country) & (df['Product'] == product))

        # Apply the mask to select only the rows where the condition is True
        self.dataframe = df[mask]

    def _filter_zeros(self, time_years):
        countries = self.get_countries()
        products = self.get_products()

        for country in countries:
            for product in products:
                t, y = self.get_time_series_last_days(product, country, time_years=time_years)
                if len(t) == 0:
                    continue
                if sum(y) == 0:
                    self.delete_rows_by_country(product, country)

    def load_data(self, time_years=1., **kwargs):
        """
        Reads the CSV file into a Pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to the pandas.read_csv() function.

        Returns:
            pandas.DataFrame or None: The loaded DataFrame if successful, None otherwise.
        """
        try:
            self.dataframe = pd.read_csv(self.file_path, **kwargs)

            # Preprocessing
            print('Preprocessing time')
            self._preprocess_time()
            print('Preprocessing zeros')
            self._filter_zeros(time_years)

            self.tensor_data = None  # Reset tensor when data is reloaded
            return self.dataframe
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{self.file_path}'")
            return None
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None

    def get_dataframe(self):
        """
        Returns the loaded Pandas DataFrame.

        Returns:
            pandas.DataFrame or None: The loaded DataFrame, or None if no data has been loaded.
        """
        return self.dataframe

    def get_countries(self):
        df = self.get_dataframe()
        return df["Country"].unique().tolist()

    def get_products(self):
        df = self.get_dataframe()
        return df["Product"].unique().tolist()

    def get_time_series(self, product, country):
        df = self.get_dataframe()
        df2 = df[df["Product"] == product]
        df3 = df2[df2["Country"] == country]
        x = df3["abs_time"].to_numpy()
        y = df3["Quantity"].to_numpy()
        return x, y

    def get_time_series_last_days(self, product, country, time_years):
        """get the last n days of the series"""
        t, y = self.get_time_series(product, country)
        t_last = t[-1]
        mask = t > t_last - time_years
        return t[mask], y[mask]

    def _one_hot_encode(self, columns_to_encode):
        """
        Performs one-hot encoding on specified columns of the DataFrame.

        Args:
            columns_to_encode (list of str, optional): A list of column names to
                                                      one-hot encode. If None, no
                                                      encoding is performed.
        """
        if self.dataframe is not None and columns_to_encode:
            try:
                self.one_hot_dataframe = pd.get_dummies(self.dataframe, columns=columns_to_encode)
            except KeyError as e:
                print(f"Error: Column '{e}' not found in the DataFrame for one-hot encoding.")
            except Exception as e:
                print(f"An error occurred during one-hot encoding: {e}")
        elif columns_to_encode:
            print("Warning: DataFrame not loaded. Call load_data() before one-hot encoding.")

    def get_one_hot_data(self, columns_to_encode):
        if self.one_hot_dataframe is None:
            self._one_hot_encode(columns_to_encode)
        return self.one_hot_dataframe

    def _to_tensor(self, dtype=torch.float32):
        """
        Converts the processed Pandas DataFrame (after optional one-hot encoding)
        to a PyTorch tensor.

        Args:
            dtype (torch.dtype, optional): The desired data type of the tensor.
                                            Defaults to torch.float32.

        Returns:
            torch.Tensor or None: The PyTorch tensor representation of the DataFrame,
                                 or None if the DataFrame is not loaded.
        """
        if self.dataframe is not None:
            try:
                # Convert DataFrame to NumPy array
                numpy_array = self.dataframe.values
                # Convert NumPy array to PyTorch tensor
                self.tensor_data = torch.tensor(numpy_array, dtype=dtype)
                return self.tensor_data
            except Exception as e:
                print(f"An error occurred during tensor conversion: {e}")
                return None
        else:
            print("Error: DataFrame not loaded. Call load_data() first.")
            return None

    def get_tensor_data(self):
        """
        Returns the PyTorch tensor representation of the loaded and processed DataFrame.

        Returns:
            torch.Tensor or None: The PyTorch tensor, or None if the DataFrame
                                 has not been loaded or converted.
        """
        if self.tensor_data is None:
            self._to_tensor()
        return self.tensor_data
