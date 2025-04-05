import pandas as pd


class CSVLoader:
    """
    A class to read data from a CSV file and load it into a Pandas DataFrame.
    """
    def __init__(self, file_path):
        """
        Initializes the CSVLoader with the path to the CSV file.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.dataframe = None

    def load_data(self, **kwargs):
        """
        Reads the CSV file into a Pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to the pandas.read_csv() function.
                       This allows for customization of how the CSV is read (e.g., specifying
                       a delimiter, header row, data types, etc.).

        Returns:
            pandas.DataFrame or None: The loaded DataFrame if successful, None otherwise.
        """
        try:
            self.dataframe = pd.read_csv(self.file_path, **kwargs)
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
