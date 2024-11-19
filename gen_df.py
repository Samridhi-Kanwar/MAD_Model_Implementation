import pandas as pd

def process_finance_data(file_path):
    # Load the Excel sheet into a DataFrame
    xls = pd.ExcelFile(file_path)
    df = xls.parse('Sheet1')

    # Clean the data by removing completely empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Extract the first column as the 'Date' column
    date_column = df.iloc[:, 0]
    df = df.iloc[:, 1:]  # Remove the first column from the main DataFrame

    # Convert the extracted date column to datetime format
    dates = pd.to_datetime(date_column, errors='coerce')

    # Separate columns into prices and returns based on the ".1" suffix
    price_columns = [col for col in df.columns if not str(col).endswith('.1')]
    returns_columns = [col for col in df.columns if str(col).endswith('.1')]

    # Create the two DataFrames
    prices_df = df[price_columns].copy()
    returns_df = df[returns_columns].copy()

    # Assign the dates to both DataFrames
    prices_df.insert(0, 'Date', dates)
    returns_df.insert(0, 'Date', dates)

    # Drop rows with NaT in the 'Date' column
    prices_df.dropna(subset=['Date'], inplace=True)
    returns_df.dropna(subset=['Date'], inplace=True)

    return prices_df[:37], returns_df[:37]
