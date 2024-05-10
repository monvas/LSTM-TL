def tl(features, data_clean, index, lag, n_ahead, n_layer, learning_rate, epochs, batch_size, source_model_path, data_sample):
    """
    A method to create a Multivariate LSTM model with TL using different portions of the data
    """
    for k,v in data_sample.items():
        separator = "=" * 130 
        print()
        print()
        print(separator)
        print(f'Transfer Learning using {k}% of the data')
        print(separator)
        print()

        # Calculate the index to start slicing from
        start_index = int(v * len(data_clean))

        # Select the last sample % of records
        data_sample = data_clean.iloc[start_index:]

        data_clean_s, train_mean_sample, train_std_sample, test_share_sample = scale_data(features, data_sample, index)
        display(data_clean_s)
        X, Y = create_X_Y(data_clean_s.values, lag=lag, n_ahead=n_ahead)
        n_ft = X.shape[2]

        # Spliting into train and test sets 
        Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share_sample))], Y[0:int(X.shape[0] * (1 - test_share_sample))]
        Xval, Yval = X[int(X.shape[0] * (1 - test_share_sample)):], Y[int(X.shape[0] * (1 - test_share_sample)):]

        print(f"Shape of training data: {Xtrain.shape}")
        print(f"Shape of the target data: {Ytrain.shape}")

        print(f"Shape of validation data: {Xval.shape}")
        print(f"Shape of the validation target data: {Yval.shape}")

        model = NNMultistepModel(
            X=Xtrain,
            Y=Ytrain,
            n_outputs=n_ahead,
            n_lag=lag,
            n_ft=n_ft,
            n_layer=n_layer,
            Xval=Xval,
            Yval=Yval,
            transfer_learning=True,
            source_model=source_model_path
        )

        model.model.summary()

        history = model.train(lr=0.01,epochs=20, batch_size=batch_size)
        training_time = model.get_training_time()

        loss = history.history.get('loss')
        val_loss = history.history.get('val_loss')
        n_epochs = range(len(loss))

        plt.figure(figsize=(9, 7))
        plt.plot(n_epochs, loss, 'r', label='Training loss', color='blue')
        if val_loss is not None:
            plt.plot(n_epochs, val_loss, 'r', label='Validation loss', color='red')
        plt.legend(loc=0)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()

        model.set_trainable_layers()
        history = model.train(lr=0.0001, epochs=epochs, batch_size=batch_size)

        training_time_ft = model.get_training_time()

        # Forecasting on all the samples in the validation set 
        forecast = model.predict(Xval)

        target_index = 0

        # Reversed Metrics
        reversed_forecast = forecast * train_std_sample[target_index] + train_mean_sample[target_index]
        reversed_Yval = Yval * train_std_sample[target_index] + train_mean_sample[target_index]

        model_name = f'{building_name}_{k}'
        total_training_time = training_time + training_time_ft

        append_model_metrics(model_name, reversed_forecast, reversed_Yval, target_index, total_training_time)
