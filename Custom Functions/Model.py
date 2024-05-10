# NNMultistepModel class, create_X_Y, and scale_data from: https://github.com/Eligijus112/Vilnius-weather-LSTM/blob/main/weather-analysis.ipynb

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses, optimizers
import time
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class NNMultistepModel():
    def __init__(
        self, 
        X, 
        Y, 
        n_outputs,
        n_lag,
        n_ft,
        n_layer,
        Xval=None,
        Yval=None,
        mask_value=-999.0,
        min_delta=0.001,
        patience=5,
        transfer_learning=False,
        source_model=None  # Path to full source model
    ):
        # Transfer learning with source model
        if transfer_learning and source_model:
            base_model = load_model(source_model)
            base_model_layers = base_model.layers[:-1]

            # Create a new model without the output layer
            new_model = Model(inputs=base_model.input, outputs=base_model_layers[-1].output)
            # Freeze layers of base model
            for layer in new_model.layers:
                layer.trainable = False
            
            new_output_layer = Dense(n_outputs)(new_model.layers[-1].output)
            self.model = Model(inputs=new_model.input, outputs=new_output_layer)

        else:
            lstm_input = Input(shape=(n_lag, n_ft))
            lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)
            output_layer = Dense(n_outputs)(lstm_layer)
            self.model = Model(inputs=lstm_input, outputs=output_layer)

        self.n_layer = n_layer
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.mask_value = mask_value
        self.min_delta = min_delta
        self.patience = patience
        self.training_time = None

    def trainCallback(self):
        return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)

    def train(self, lr, epochs, batch_size):
        # Compiling the model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizers.Adam(learning_rate=self.lr))

        if (self.Xval is not None) and (self.Yval is not None):
            start_time = time.time()
            history = self.model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch_size, 
                validation_data=(self.Xval, self.Yval), 
                shuffle=False,
                callbacks=[self.trainCallback()]
            )
            end_time = time.time()
            self.training_time = end_time - start_time
            print("Training time:", self.training_time, "seconds")
        else:
            start_time = time.time()
            history = self.model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch_size,
                shuffle=False,
                callbacks=[self.trainCallback()]
            )
            end_time = time.time()
            self.training_time = end_time - start_time
            print("Training time:", self.training_time, "seconds")
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)

    def set_trainable_layers(self):
        for layer in self.model.layers:
            layer.trainable = True

    def get_training_time(self):
        return self.training_time

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an RNN input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

def append_model_metrics(model_name, reversed_forecast, reversed_Yval, target_index, training_time, csv_filename='model_metrics.csv'):
    """
    A method to write RMSE, MAE, R2 and time metrics to a csv file for each model 
    """
    # Reversed Metrics
    reversed_rmse = mean_squared_error(reversed_Yval, reversed_forecast, squared=False)
    reversed_mae = mean_absolute_error(reversed_Yval, reversed_forecast)
    reversed_r2 = r2_score(reversed_Yval, reversed_forecast)
    training_time = training_time

    print("Reversed RMSE:", reversed_rmse)
    print("Reversed MAE:", reversed_mae)
    print("Reversed R2 Score:", reversed_r2)
    print("Training Time:", training_time, "seconds")

    # List of metrics
    metrics_data = [
        {"Model": model_name, "RMSE": reversed_rmse, "MAE": reversed_mae, "R2": reversed_r2, "Time": training_time},
    ]

    # Define the fieldnames for the CSV file
    fieldnames = ["Model", "RMSE", "MAE", "R2", "Time"]

    # Append metrics to the CSV file
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        # Write metrics data
        for row in metrics_data:
            writer.writerow(row)

def scale_data(features, dataset, index):
    """
    A method to scale the data and split into train and test sets
    """
    dataset = dataset[features]

    test_days_df   = dataset[(dataset.index >= index)]
    test_share = len(test_days_df)/len(dataset)
    print(f'Test Share: {test_share}')

    nrows = dataset.shape[0]

    # Spliting into train and test sets
    train = dataset[0:int(nrows * (1 - test_share))]
    test = dataset[int(nrows * (1 - test_share)):]

    # Scaling the data 
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    # Creating the final scaled frame 
    dataset_s = pd.concat([train, test])

    return dataset_s, train_mean, train_std, test_share
