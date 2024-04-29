import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

#This is a good format for an all in one python program. For the meantime, test each function separately.

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Example: Fill missing values
    df.fillna(method='ffill', inplace=True)
    return df

def calculate_returns(df, periods):
    for period in periods:
        df[f'return_{period}d'] = df['Close'].pct_change(periods=period).fillna(0)
    return df

def calculate_macd(df, span1=12, span2=26, signal=9):
    exp1 = df['Close'].ewm(span=span1, adjust=False).mean()
    exp2 = df['Close'].ewm(span=span2, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def calculate_volatility(df, window=10):
    df['volatility'] = df['return_1d'].rolling(window=window).std()
    return df

def clean_types(df):
    #Clean date types as well if using a regression algorithm
    # Convert 'Close' column to numeric, coercing errors
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # You might have NaNs where there were non-numeric strings, fill them if necessary
    df['Close'].fillna(method='ffill', inplace=True)  # Forward fill to handle NaNs, or choose another method

    # Convert 'Date' column to datetime format
    #df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Now you can extract datetime components
    #df['Year'] = df['Date'].dt.year
    #df['Month'] = df['Date'].dt.month
    #df['Day'] = df['Date'].dt.day
    #df['day_of_week'] = df['Date'].dt.dayofweek
    # Now drop the original 'Date' column
    #df.drop('Date', axis=1, inplace=True)


    # Explicitly convert to string, replace commas, and convert to float
    def clean_numeric(column):
        # Convert to string first to ensure .str operations can be performed
        column = column.astype(str)
        # Replace commas and convert to float
        column = column.str.replace(',', '')
        return pd.to_numeric(column, errors='coerce')

    # Apply this function to the columns known to contain numeric strings with commas
    columns_to_clean = ['Open','High','Low']  # Extend this list based on your actual data
    for col in columns_to_clean:
        df[col] = clean_numeric(df[col])


    #Convert
    def convert_volume(vol):
        # Ensure vol is a string before checking for 'K' or 'M'
        vol = str(vol)
        if vol == '-' or vol == 'nan':  # Handle both '-' and unexpected NaN values represented as strings
            return np.nan
        elif 'K' in vol:
            return float(vol.replace('K', '')) * 1e3
        elif 'M' in vol:
            return float(vol.replace('M', '')) * 1e6
        else:
            try:
                return float(vol)  # Handle numbers as strings or normal numbers
            except ValueError:
                return np.nan  # In case of any other unexpected string that cannot be converted

    # Apply this function to the 'Vol.' column
    df['Vol.'] = df['Vol.'].apply(convert_volume)

    # Option to fill NaN values, choose one:
    # df['Vol.'] = df['Vol.'].fillna(0)  # Replace NaN with 0
    df['Vol.'] = df['Vol.'].fillna(df['Vol.'].mean())  # Replace NaN with mean


    #Drop the % sign from 'change %' and convert to float

    # Convert entire 'Change %' column to string, handling potential NaN values
    df['Change %'] = df['Change %'].astype(str)

    # Strip the '%' character and convert to float, dividing by 100 to turn into a decimal
    df['Change %'] = df['Change %'].str.rstrip('%').astype('float') / 100.0

    # Check for entries that were originally NaN and set them back to NaN if they were converted to strings
    df['Change %'] = df['Change %'].replace(-0.01, np.nan) 
    
    return df

def feature_engineering(df):
    periods = [1, 21, 63]  # Daily, monthly, quarterly returns
    df = calculate_returns(df, periods)
    df = calculate_macd(df)
    df = calculate_volatility(df)
    return df


#Now normalize the features
def normalize_features(df, target_column):
    scaler = StandardScaler()
    #Before doing Train Test Split, we need to add a 'Target' Column which captures the daily returns

    # Shift the 'Price' column by -1 to compare the next day's closing price to the current day
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    #Drop the last row where future price is unknown due to shift
    df.dropna(subset=['Target'], inplace=True)
    
    #if 'Date' in df.columns:
        #df['Date'] = pd.to_datetime(df['Date'])
        #df['Month'] = df['Date'].dt.month
        #df['Day'] = df['Date'].dt.day
        #df['Weekday'] = df['Date'].dt.weekday
        #df.drop('Date', axis=1, inplace=True)  # Optional: remove if not needed for modeling

    #Select only numerical columns and exclude the target column for scaling
    features = df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64')]
    features = (features[features != 'Target'])# & (features[features != 'Date'])   # Exclude the target column from scaling

    df[features] = scaler.fit_transform(df[features])
    df = df.dropna() #Drop NaN
    df = df.iloc[:-1] ##Drop the last row where future price is unknown due to shift
    return df, scaler

def prepare_data(filepath):
    df = load_data(filepath)
    df = clean_data(df)
    df = clean_types(df)
    df = feature_engineering(df)
    df, scaler = normalize_features(df)
    return df, scaler

def train_test_split_data(df, test_size=0.2, random_state=42):
    X = df.drop('target', axis=1)
    y = df['target']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)    
    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    filepath = 'path_to_your_data.csv'
    df, scaler = prepare_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    ## Optionally save the processed data
    X_train.to_csv('train_features.csv')
    y_train.to_csv('train_labels.csv')
    X_test.to_csv('test_features.csv')
    y_test.to_csv('test_labels.csv')
    X_val.to_csv('val_features.csv')
    y_val.to_csv('val_labels.csv')