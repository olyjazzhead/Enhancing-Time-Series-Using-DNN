import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(df):
    df = pd.read_csv('path_to_training_data.csv')
    return df
    
def drop_date(df):
    #Date = False for Classification, Date=True for Regression
    Date = False
    if Date == True:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['Year'] = df['Date'].dt.day
            df['Weekday'] = df['Date'].dt.weekday
            df.drop('Date', axis=1, inplace=True)
    else:
        df.drop('Date', axis=1, inplace=True)  # Optional: remove if not needed for modeling
    return df

def begin_train(df):
    df = df.sample(frac=1).reset_index(drop=True)  #Shuffle the dataset
    target = df["Target"]   #Delcare Target
    data=pd.DataFrame(df, columns=['Close', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'return_1d','macd', 'macd_signal', 'macd_hist',
       'volatility'])   # For simplicity only keep just 1_d returns. Can add more later
    print ("Df shape of dataset to be used :",data.shape)
    display(data.head())
    display(target.head())
    return df


def train_test_split(df):
    X_temp, X_test, y_temp, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    #Basically just re-adding the original element indexing from pandas
    y_train, y_test = y_train.reset_index(drop=True),y_test.reset_index(drop=True)  #Reset index for dataseries, not needed for ndarray (X_train, X_test)
    print ("X_train shape:",X_train.shape)
    print ("y_train shape:",y_train.shape)
    print ("X_test shape:",X_test.shape)
    print ("y_test shape:",y_test.shape)
    return X_train, X_test, X_val, y_train, y_test, y_val
    
def build_model(df):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.initializers import HeNormal
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.optimizers import Adam


    #Add custom loss function
    def sharpe_loss(y_true, y_pred):
        # Assuming y_pred are returns predictions and y_true are actual returns
        # Calculate expected return and volatility (standard deviation)
        returns = tf.reduce_mean(y_pred)  # Expected return
        volatility = tf.math.reduce_std(y_pred)  # Standard deviation of returns
        sharpe_ratio = returns / (volatility + 1e-6)  # Added a small constant to avoid division by zero
        return -sharpe_ratio  # Minimize the negative Sharpe ratio (i.e., maximize the Sharpe ratio)



    #Construct model

    input_shape=X_train.shape[0]

    def build_model(input_shape):
        model = Sequential([
            Dense(64, activation='relu', kernel_initializer=HeNormal(), input_shape=(input_shape,)),
            Dropout(0.1),
            Dense(64, activation='relu', kernel_initializer=HeNormal()),
            Dropout(0.1),
            Dense(1, activation='linear')  # Assuming a binary classification problem
        ])
        model.compile(optimizer=Adam(learning_rate=.0001), loss=sharpe_loss, metrics=['mean_squared_error'])
        return model

    model = build_model(input_shape)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val))

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def train_model(df):
    df = load_data(df)
    df = drop_date(df)
    df = begin_train(df)
    return df
    

def run_model(df):
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(df)
    build_model(df)
    return df
    
if __name__ == "__main__":
    df = train_model(df)
    run_model(df)